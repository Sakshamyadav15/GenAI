/** Balance-scales mark. Inline SVG, so no image download is needed. Colour follows `currentColor`. */
export function ScalesMark({ className = 'h-8 w-8' }: { className?: string }) {
  return (
    <svg viewBox="0 0 32 32" className={className} fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M16 6v20" />
      <path d="M10 27h12" />
      <path d="M5 9.5h22" />
      <circle cx="16" cy="4.6" r="1.4" fill="currentColor" stroke="none" />
      <path d="M5 9.5 1.8 17.5h6.4z" />
      <path d="M1.8 17.5a3.2 2.4 0 0 0 6.4 0" />
      <path d="M27 9.5l-3.2 8h6.4z" />
      <path d="M23.8 17.5a3.2 2.4 0 0 0 6.4 0" />
    </svg>
  );
}
