import { Textarea } from "@/components/ui/textarea";
import { Button } from "@/components/ui/button";
import { Loader2 } from "lucide-react";

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
  return (
    <div className="space-y-3">
      <Textarea
        placeholder="Entrez votre texte ici…"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="min-h-[120px]"
      />

      <Button onClick={onSend} disabled={loading} className="w-full">
        {loading ? (
          <span className="flex items-center gap-2">
            <Loader2 className="animate-spin" size={16} />
            Génération…
          </span>
        ) : (
          "Envoyer à Mnemos"
        )}
      </Button>
    </div>
  );
}
