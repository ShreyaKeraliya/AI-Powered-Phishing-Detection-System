import Link from "next/link";
import UrlForm from "../components/UrlForm";

export default function UrlPage() {
  return (
    <main className="container">
      <nav className="top-nav">
        <Link href="/">← Dashboard</Link>
      </nav>
      <UrlForm />
    </main>
  );
}


