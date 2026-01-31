"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { motion } from "framer-motion";
import { Sparkles, Bot } from "lucide-react";
import PromptInput from "@/components/PromptInput";
import ResponseBox from "@/components/ResponseBox";
import GithubButton from "@/components/GithubButton";
import { fetchChatResponse } from "@/services/mnemosApi";

export default function MnemosChat() {
  const [prompt, setPrompt] = useState("");
  const [response, setResponse] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSend = async () => {
    if (!prompt.trim()) return;

    setLoading(true);
    setError(null);
    setResponse("");

    try {
      const output = await fetchChatResponse({ prompt });
      setResponse(output.text);
    } catch (err: any) {
      setError(err.message ?? "Erreur inconnue");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="relative min-h-screen flex items-center justify-center bg-gradient-to-br from-slate-50 via-gray-100 to-slate-200 p-4 font-sans text-slate-800">

      <GithubButton />

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, ease: "easeOut" }}
        className="w-full max-w-2xl z-10"
      >
        <Card className="rounded-2xl shadow-xl border border-white/50 bg-white/80 backdrop-blur-sm overflow-hidden">
          
          <CardHeader className="border-b border-gray-100 bg-white/50 pb-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
            <div className="w-10 flex items-center justify-center">
                <img 
                src="/mnemos.png" 
                alt="Mnemos Logo" 
                className="w-full h-auto object-contain rounded-lg"
                />
            </div>
                <div>
                    <CardTitle className="text-xl font-bold text-gray-900 tracking-tight">
                    Mnemos
                    </CardTitle>
                    <p className="text-xs text-gray-500 font-medium">Mini Transformer Model</p>
                </div>
              </div>
              
              <div className="px-3 py-1 rounded-full bg-indigo-50 border border-indigo-100 text-indigo-700 text-xs font-semibold flex items-center gap-1">
                <Sparkles size={12} />
                v0.1
              </div>
            </div>
          </CardHeader>

          <CardContent className="p-6 space-y-6">
            <ResponseBox response={response} error={error} />
            
            <PromptInput
              value={prompt}
              onChange={setPrompt}
              onSend={handleSend}
              loading={loading}
            />
          </CardContent>
        </Card>
        
        <div className="mt-4 text-center text-xs text-gray-400">
          Généré par Mnemos-0.1 • Le modèle peut (et va) faire des erreurs.
        </div>
      </motion.div>
    </div>
  );
}