export function BrandLogo({ title = "pgpilot forecast", subtitle = "termikk og vind for paraglidere" }) {
  return (
    <div className="brand-wrap">
      <h1 className="brand-shell" aria-label={title}>
        <span aria-hidden className="brand-shadow">
          {title}
        </span>
        <span className="brand-main">{title}</span>
      </h1>
      <p className="brand-tagline">{subtitle}</p>
    </div>
  );
}
