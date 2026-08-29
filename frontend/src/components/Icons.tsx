/** Inline SVG icon set — no icon-library dependency. */

interface IconProps {
  size?: number;
  className?: string;
}

const base = (size: number) => ({
  width: size,
  height: size,
  viewBox: '0 0 24 24',
  fill: 'none',
  stroke: 'currentColor',
  strokeWidth: 2,
  strokeLinecap: 'round' as const,
  strokeLinejoin: 'round' as const,
});

export const IconLogo = ({ size = 18, className }: IconProps) => (
  <svg {...base(size)} className={className} aria-hidden="true">
    <path d="M4 5a2 2 0 0 1 2-2h8l4 4v12a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2z" />
    <path d="M14 3v4h4" />
    <circle cx="11" cy="13" r="2.5" />
    <path d="m14 16 2.5 2.5" />
  </svg>
);

export const IconUpload = ({ size = 18, className }: IconProps) => (
  <svg {...base(size)} className={className} aria-hidden="true">
    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
    <path d="m7 10 5-5 5 5" />
    <path d="M12 5v12" />
  </svg>
);

export const IconTrash = ({ size = 15, className }: IconProps) => (
  <svg {...base(size)} className={className} aria-hidden="true">
    <path d="M3 6h18" />
    <path d="M8 6V4a1 1 0 0 1 1-1h6a1 1 0 0 1 1 1v2" />
    <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6" />
  </svg>
);

export const IconCheck = ({ size = 12, className }: IconProps) => (
  <svg {...base(size)} strokeWidth={3} className={className} aria-hidden="true">
    <path d="m5 12 5 5L20 7" />
  </svg>
);

export const IconSend = ({ size = 17, className }: IconProps) => (
  <svg {...base(size)} className={className} aria-hidden="true">
    <path d="m4 4 16 8-16 8 3-8z" />
  </svg>
);

export const IconStop = ({ size = 15, className }: IconProps) => (
  <svg
    width={size}
    height={size}
    viewBox="0 0 24 24"
    fill="currentColor"
    className={className}
    aria-hidden="true"
  >
    <rect x="6" y="6" width="12" height="12" rx="2" />
  </svg>
);

export const IconSun = ({ size = 17, className }: IconProps) => (
  <svg {...base(size)} className={className} aria-hidden="true">
    <circle cx="12" cy="12" r="4" />
    <path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" />
  </svg>
);

export const IconMoon = ({ size = 17, className }: IconProps) => (
  <svg {...base(size)} className={className} aria-hidden="true">
    <path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z" />
  </svg>
);

export const IconRefresh = ({ size = 13, className }: IconProps) => (
  <svg {...base(size)} className={className} aria-hidden="true">
    <path d="M21 12a9 9 0 1 1-3-6.7L21 8" />
    <path d="M21 3v5h-5" />
  </svg>
);

export const IconCopy = ({ size = 13, className }: IconProps) => (
  <svg {...base(size)} className={className} aria-hidden="true">
    <rect x="9" y="9" width="12" height="12" rx="2" />
    <path d="M5 15V5a2 2 0 0 1 2-2h10" />
  </svg>
);

export const IconDownload = ({ size = 15, className }: IconProps) => (
  <svg {...base(size)} className={className} aria-hidden="true">
    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
    <path d="m7 10 5 5 5-5" />
    <path d="M12 3v12" />
  </svg>
);

export const IconSearch = ({ size = 15, className }: IconProps) => (
  <svg {...base(size)} className={className} aria-hidden="true">
    <circle cx="11" cy="11" r="7" />
    <path d="m20 20-3.5-3.5" />
  </svg>
);

export const IconChevron = ({ size = 13, className }: IconProps) => (
  <svg {...base(size)} className={className} aria-hidden="true">
    <path d="m6 9 6 6 6-6" />
  </svg>
);

export const IconQuote = ({ size = 13, className }: IconProps) => (
  <svg {...base(size)} className={className} aria-hidden="true">
    <path d="M9 5H5a2 2 0 0 0-2 2v4h6V5zM19 5h-4a2 2 0 0 0-2 2v4h6V5z" />
    <path d="M3 11c0 5 2 7 6 8M13 11c0 5 2 7 6 8" />
  </svg>
);

export const IconSparkle = ({ size = 15, className }: IconProps) => (
  <svg {...base(size)} className={className} aria-hidden="true">
    <path d="M12 3l1.9 5.6L19.5 10l-5.6 1.9L12 17.5l-1.9-5.6L4.5 10l5.6-1.4z" />
    <path d="M18.5 16.5l.7 2 2 .7-2 .7-.7 2-.7-2-2-.7 2-.7z" />
  </svg>
);

export const IconAlert = ({ size = 16, className }: IconProps) => (
  <svg {...base(size)} className={className} aria-hidden="true">
    <circle cx="12" cy="12" r="9" />
    <path d="M12 8v4M12 16h.01" />
  </svg>
);

export const IconClose = ({ size = 14, className }: IconProps) => (
  <svg {...base(size)} className={className} aria-hidden="true">
    <path d="M18 6 6 18M6 6l12 12" />
  </svg>
);

export const IconMenu = ({ size = 18, className }: IconProps) => (
  <svg {...base(size)} className={className} aria-hidden="true">
    <path d="M3 6h18M3 12h18M3 18h18" />
  </svg>
);

export const IconSpinner = ({ size = 15, className }: IconProps) => (
  <svg
    {...base(size)}
    className={`spin ${className ?? ''}`}
    aria-hidden="true"
  >
    <path d="M21 12a9 9 0 1 1-6.2-8.6" />
  </svg>
);

export const IconFile = ({ size = 15, className }: IconProps) => (
  <svg {...base(size)} className={className} aria-hidden="true">
    <path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z" />
    <path d="M14 3v5h5" />
  </svg>
);
