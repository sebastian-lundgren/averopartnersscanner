import Link from "next/link";

const links = [
  ["/", "Dashboard"],
  ["/upload", "Opplasting"],
  ["/library", "Bildebibliotek"],
  ["/review", "Review-kø"],
  ["/annotations", "Annotering / læring"],
  ["/addresses", "Adresser"],
  ["/models", "Modellversjoner"],
  ["/export", "Eksport"],
  ["/scanner", "Street View-scan"],
  ["/settings", "Innstillinger"],
];

export default function Nav() {
  return (
    <nav>
      <span className="brand">Alarmskilt QC</span>
      {links.map(([href, label]) => (
        <Link key={href} href={href}>
          {label}
        </Link>
      ))}
    </nav>
  );
}
