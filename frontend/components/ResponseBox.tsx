import { AlertCircle, Bot } from "lucide-react";

interface Props {
  response: string;
  error: string | null;
}

export default function ResponseBox({ response, error }: Props) {
  if (error) {
    return (
        <div className="p-4 rounded-xl bg-red-50 border border-red-100 text-red-600 flex items-start gap-3 text-sm animate-in fade-in slide-in-from-bottom-2">
            <AlertCircle size={18} className="mt-0.5" />
            <p>{error}</p>
        </div>
    );
  }
  
  if (!response) {
      return (
        <div className="h-32 flex flex-col items-center justify-center text-gray-300 border-2 border-dashed border-gray-100 rounded-xl">
            <Bot size={32} className="mb-2 opacity-50" />
            <p className="text-sm">En attente de votre prompt...</p>
        </div>
      );
  }

  return (
    <div className="bg-slate-50 border border-slate-100 rounded-xl p-5 shadow-inner animate-in fade-in zoom-in-95 duration-300">
      <div className="flex items-center gap-2 mb-3 pb-2 border-b border-slate-200/60">
        <span className="text-xs font-bold text-indigo-600 uppercase tracking-wider">Réponse</span>
      </div>
      <div className="prose prose-sm max-w-none text-gray-700 leading-relaxed whitespace-pre-wrap">
        {response}
      </div>
    </div>
  );
}