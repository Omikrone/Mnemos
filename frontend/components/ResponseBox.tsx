interface Props {
  response: string;
  error: string | null;
}

export default function ResponseBox({ response, error }: Props) {
  if (error) {
    return <p className="text-sm text-red-600">{error}</p>;
  }

  if (!response) return null;

  return (
    <div className="bg-gray-100 rounded-xl p-4 whitespace-pre-wrap">
      <h2 className="font-medium mb-2">Réponse du modèle</h2>
      <p className="text-sm">{response}</p>
    </div>
  );
}