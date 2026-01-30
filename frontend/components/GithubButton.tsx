import { Github } from "lucide-react";

export default function GithubButton() {
  return (
    <a
      href="https://github.com/Omikrone/Mnemos"
      target="_blank"
      rel="noopener noreferrer"
      className="absolute top-4 right-4 z-50 inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-gray-900 text-white hover:bg-gray-800 transition-all hover:scale-105 shadow-md text-sm font-medium"
    >
      <Github size={18} />
      <span className="hidden sm:inline">Voir sur GitHub</span>
    </a>
  );
}