"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";

export default function Navbar() {
  const pathname = usePathname();
  const [open, setOpen] = useState(false);

  const linkStyle = (path) => ({
    color: pathname === path ? "#38bdf8" : "inherit",
    fontWeight: pathname === path ? 600 : 400,
    borderBottom:
      pathname === path ? "2px solid #38bdf8" : "none",
    paddingBottom: "4px",
  });

  return (
    <nav className="navbar">
      <div className="nav-wrapper">
        <Link href="/" className="logo">
          Phishing Detection AI
        </Link>

        {/* Desktop Nav */}
        <div className="desktop-nav">
          <Link href="/" style={linkStyle("/")}>Home</Link>
          <Link href="/email" style={linkStyle("/email")}>Email Detection</Link>
          <Link href="/url" style={linkStyle("/url")}>URL Detection</Link>
          <Link href="/awareness" style={linkStyle("/awareness")}>Awareness</Link>
          <Link href="/about" style={linkStyle("/about")}>About</Link>
        </div>

        {/* Mobile Button */}
        <button
          className="menu-btn"
          onClick={() => setOpen(!open)}
        >
          ☰
        </button>
      </div>

      {/* Mobile Menu */}
      {open && (
        <div className="mobile-nav">
          <Link href="/" onClick={() => setOpen(false)}>Home</Link>
          <Link href="/email" onClick={() => setOpen(false)}>Email Detection</Link>
          <Link href="/url" onClick={() => setOpen(false)}>URL Detection</Link>
          <Link href="/awareness" onClick={() => setOpen(false)}>Awareness</Link>
          <Link href="/about" onClick={() => setOpen(false)}>About</Link>
        </div>
      )}
    </nav>
  );
}
