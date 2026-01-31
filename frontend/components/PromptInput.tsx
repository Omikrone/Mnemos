import { Textarea } from "@/components/ui/textarea";
import { Button } from "@/components/ui/button";
import { Loader2, Send } from "lucide-react";

interface Props {
  value: string;
  onChange: (v: string) => void;
  onSend: () => void;
  loading: boolean;
}

export default function PromptInput({
  value,
  onChange,
  onSend,
  loading,
}: Props) {
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && e.ctrlKey && !loading) {
        onSend();
    }
  };

  return (
    <div className="space-y-3">
      <div className="relative">
        <Textarea
          placeholder="Posez votre question à Mnemos..."
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={handleKeyDown}
          className="min-h-[100px] resize-none pr-4 py-3 text-base rounded-xl border-gray-200 focus:border-indigo-400 focus:ring-indigo-100 transition-all shadow-sm bg-white"
        />
        <div className="absolute bottom-2 right-2 text-xs text-gray-400 pointer-events-none">
            Ctrl + Enter
        </div>
      </div>

      <Button 
        onClick={onSend} 
        disabled={loading || !value.trim()} 
        className="w-full rounded-xl bg-indigo-600 hover:bg-indigo-700 text-white font-medium py-6 transition-all shadow-md hover:shadow-lg active:scale-[0.99]"
      >
        {loading ? (
          <span className="flex items-center gap-2">
            <Loader2 className="animate-spin" size={18} />
            Génération en cours...
          </span>
        ) : (
          <span className="flex items-center gap-2">
            Envoyer à Mnemos <Send size={16} />
          </span>
        )}
      </Button>
    </div>
  );
}