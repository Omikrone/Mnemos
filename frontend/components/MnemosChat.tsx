"use client";


import { useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { motion } from "framer-motion";
import PromptInput from "@/components/PromptInput"
import ResponseBox from "@/components/ResponseBox"
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
            const output = await fetchChatResponse({prompt});
            setResponse(output.text);
        } catch (err: any) {
            setError(err.message ?? "Erreur inconnue");
        } finally {
            setLoading(false);
        }
    };


    return (
        <div className="min-h-screen flex items-center justify-center bg-gray-50 p-4">
            <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4 }}
            className="w-full max-w-2xl"
            >
                <Card className="rounded-2xl shadow-lg">
                    <CardContent className="p-6 space-y-4">
                        <h1 className="text-2xl font-semibold">Mnemos – Interface Web</h1>


                        <PromptInput
                            value={prompt}
                            onChange={setPrompt}
                            onSend={handleSend}
                            loading={loading}
                        />


                        <ResponseBox response={response} error={error} />
                    </CardContent>
                </Card>
            </motion.div>
        </div>
    );
}