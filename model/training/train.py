import math
import pickle
import sys
import numpy as np
import json
from datetime import datetime
import time

from metrics.training_logger import TrainingLogger
from model.transformer_model import TransformerModel
from training.tokenizer.parallel_encoding import tokenize_text
from training.batch import BatchBuilder
from training.cross_entropy import CrossEntropyLoss
from training.preprocesser import PreProcesser
from training.tokenizer.bpe import BPETokenizer
from config.paths import TRAINING_DATA_PATH, MODEL_PATH, VOCABULARY_PATH
from config.params import LEARNING_RATE, NB_EPOCHS

class Trainer:
    """Trainer class with enhanced logging and monitoring."""

    def __init__(self, resume_training=False):
        self.preprocesser = PreProcesser()
        
        self.loss_fn = CrossEntropyLoss()
        self.batch_times = []
        self.logger = TrainingLogger()
        self.start_time = None
        
        if resume_training and MODEL_PATH.exists():
            print("Resuming training from last checkpoint...")
            self.tokenizer = BPETokenizer(self.preprocesser(TRAINING_DATA_PATH))
            with open(MODEL_PATH, "rb") as f:
                model_params = pickle.load(f)
            self.model = TransformerModel.from_params(model_params)
            self.best_val_loss = float('inf')
        else:
            self.tokenizer = self._initialize_tokenizer()
            self.model = TransformerModel(self.tokenizer.get_vocabulary_size())
            self.best_val_loss = float('inf')

    def _initialize_tokenizer(self) -> BPETokenizer:
        """Initialize tokenizer with memory-efficient processing."""
        if not TRAINING_DATA_PATH.exists():
            raise FileNotFoundError(f"Training data not found at {TRAINING_DATA_PATH}")
        
        cleaned_data = self.preprocesser(TRAINING_DATA_PATH)
        tokenizer = BPETokenizer(cleaned_data)
        tokenizer.build()
        return tokenizer

    def _lr_scheduler(self, step: int, warmup_steps: int, total_steps: int) -> float:
        """Learning rate scheduler with cosine decay."""
        if step < warmup_steps:
            return LEARNING_RATE * (step / warmup_steps)
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return LEARNING_RATE * (0.5 * (1 + math.cos(math.pi * progress)))

    def _run_batch(self, batch: tuple, is_training: bool) -> float:
        """Process batch and return loss with timing."""
        start_time = time.time()
        input_ids, targets = batch
        logits = self.model.forward(input_ids, train=is_training)
        loss = self.loss_fn(logits, targets)
        
        if is_training:
            loss_gradient = self.loss_fn.backward(logits, targets)
            self.model.backward(loss_gradient)
            self.model.step(self.lr)
            self.model.zero_grad()
        
        batch_time = time.time() - start_time
        self.batch_times.append(batch_time)
        return float(loss)

    def _get_time_estimate(self, batches_remaining: int) -> str:
        """Calculate estimated time remaining."""
        if not self.batch_times:
            return "N/A"
        avg_time = np.mean(self.batch_times[-10:])
        total_seconds = int(avg_time * batches_remaining)
        return f"{total_seconds//3600}h {(total_seconds%3600)//60}m {total_seconds%60}s"

    def train(self):
        """Enhanced training loop with detailed logging."""
        self.start_time = time.time()
        print(f"\n🚀 Starting training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Batch preparation
        all_tokens = tokenize_text(self.tokenizer.text)
        batch_builder = BatchBuilder(self.tokenizer.text, self.tokenizer)
        chunks = batch_builder.create_chunks(all_tokens)
        train_batches, val_batches = batch_builder.create_batches(chunks)
        
        total_batches = len(train_batches)
        warmup_steps = total_batches // 4
        total_steps = total_batches * NB_EPOCHS
        
        print(f"📊 Dataset stats: {len(train_batches)} train batches | {len(val_batches)} val batches")
        print(f"⚙️ Training config: {NB_EPOCHS} epochs | LR: {LEARNING_RATE} | Warmup: {warmup_steps} steps")

        for epoch in range(1, NB_EPOCHS + 1):
            print(f"\n🌈 Epoch {epoch}/{NB_EPOCHS}")
            np.random.shuffle(train_batches)
            
            epoch_train_loss = []
            epoch_val_loss = []
            batch_logs = []
            
            # Training phase
            for i, batch in enumerate(train_batches, 1):
                global_step = (epoch-1)*total_batches + i
                self.lr = self._lr_scheduler(global_step, warmup_steps, total_steps)
                
                # Process batch
                loss = self._run_batch(batch, is_training=True)
                epoch_train_loss.append(loss)
                
                # Validation (10% of epoch)
                if i % max(1, total_batches//10) == 0 or i == total_batches:
                    val_batch = val_batches[np.random.randint(len(val_batches))]
                    val_loss = self._run_batch(val_batch, is_training=False)
                    epoch_val_loss.append(val_loss)
                    
                    # Model checkpointing
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self._save_model()
                        print(f"\n💾 New best model saved (val_loss={val_loss:.4f})")
                
                # Enhanced logging every 2% of epoch or every batch if small
                log_freq = max(1, total_batches//50)
                if i % log_freq == 0 or i == total_batches:
                    # Calculate metrics
                    avg_train = np.mean(epoch_train_loss[-log_freq*2:] or [0])
                    avg_val = np.mean(epoch_val_loss) if epoch_val_loss else 0
                    batches_remaining = (NB_EPOCHS - epoch)*total_batches + (total_batches - i)
                    eta = self._get_time_estimate(batches_remaining)
                    
                    # Detailed log entry
                    log_entry = {
                        'epoch': epoch,
                        'batch': i,
                        'total_batches': total_batches,
                        'train_loss': avg_train,
                        'val_loss': avg_val,
                        'lr': self.lr,
                        'time_per_batch': np.mean(self.batch_times[-10:]),
                        'eta': eta
                    }
                    batch_logs.append(log_entry)
                    
                    # Console output
                    progress = (i/total_batches)*100
                    sys.stdout.write(
                        f"\r🔍 Batch {i}/{total_batches} ({progress:.1f}%) | "
                        f"Train: {avg_train:.4f} | Val: {avg_val:.4f} | "
                        f"LR: {self.lr:.2e} | ETA: {eta}"
                    )
                    sys.stdout.flush()
            
            # Epoch summary
            epoch_time = time.time() - self.start_time
            avg_train_loss = np.mean(epoch_train_loss)
            avg_val_loss = np.mean(epoch_val_loss) if epoch_val_loss else 0
            
            print(f"\n\n📈 Epoch {epoch} Summary:")
            print(f"⏱️  Duration: {epoch_time//60:.0f}m {epoch_time%60:.0f}s")
            print(f"📉 Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
            print(f"🏆 Best Val Loss: {self.best_val_loss:.4f}")
            
            # Log all batch details
            for log in batch_logs:
                self.logger.train_log(
                    epoch=log['epoch'],
                    step=log['batch'],
                    loss=log['train_loss'],
                    perplexity=np.exp(log['train_loss']) if log['train_loss'] else 0
                )

        # Final save and summary
        training_time = time.time() - self.start_time
        print(f"\n🎉 Training completed in {training_time//3600:.0f}h {(training_time%3600)//60:.0f}m!")
        print(f"🏁 Final Best Validation Loss: {self.best_val_loss:.4f}")
        self._save_model()

    def _save_model(self):
        """Atomic model saving with backup."""
        try:
            model_params = self.model.get_parameters()
            vocab = self.tokenizer.table_manager.load_table()
            
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            VOCABULARY_PATH.parent.mkdir(parents=True, exist_ok=True)
            
            # Create temporary files
            temp_model = MODEL_PATH.with_suffix('.tmp')
            temp_vocab = VOCABULARY_PATH.with_suffix('.tmp')
            
            # Save to temporary files
            with open(temp_model, 'wb') as f:
                pickle.dump(model_params, f)
            with open(temp_vocab, 'w') as f:
                json.dump(vocab, f, indent=4)
            
            # Atomic rename
            temp_model.replace(MODEL_PATH)
            temp_vocab.replace(VOCABULARY_PATH)
            
            # Keep backup of previous best model
            if MODEL_PATH.exists():
                backup_path = MODEL_PATH.with_name(f"backup_{int(time.time())}.pkl")
                MODEL_PATH.replace(backup_path)
        except Exception as e:
            print(f"⚠️ Error saving model: {str(e)}")
