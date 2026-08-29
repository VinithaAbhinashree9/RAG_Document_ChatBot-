/** Chat input: auto-growing textarea, scope indicator, send/stop control. */

import { useEffect, useRef } from 'react';
import { IconSend, IconStop } from './Icons';

interface ComposerProps {
  value: string;
  disabled: boolean;
  streaming: boolean;
  scopeLabel: string;
  onChange: (value: string) => void;
  onSubmit: () => void;
  onStop: () => void;
}

const MAX_HEIGHT = 168;

export function Composer({
  value,
  disabled,
  streaming,
  scopeLabel,
  onChange,
  onSubmit,
  onStop,
}: ComposerProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Grow the textarea with its content, up to a ceiling.
  useEffect(() => {
    const element = textareaRef.current;
    if (!element) return;
    element.style.height = 'auto';
    element.style.height = `${Math.min(element.scrollHeight, MAX_HEIGHT)}px`;
  }, [value]);

  const canSend = value.trim().length > 0 && !disabled && !streaming;

  return (
    <div className="composer">
      <div className="composer__inner">
        <div className="composer__scope">
          <span>Answering from:</span>
          <span className="badge badge--accent">{scopeLabel}</span>
        </div>

        <div className="composer__field">
          <textarea
            ref={textareaRef}
            rows={1}
            value={value}
            disabled={disabled}
            placeholder={
              disabled
                ? 'Upload a document to start asking questions…'
                : 'Ask a question about your documents…'
            }
            onChange={(event) => onChange(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                if (canSend) onSubmit();
              }
            }}
            aria-label="Your question"
          />

          {streaming ? (
            <button
              type="button"
              className="send-btn send-btn--stop"
              onClick={onStop}
              title="Stop generating"
              aria-label="Stop generating"
            >
              <IconStop />
            </button>
          ) : (
            <button
              type="button"
              className="send-btn"
              onClick={onSubmit}
              disabled={!canSend}
              title="Send question"
              aria-label="Send question"
            >
              <IconSend />
            </button>
          )}
        </div>

        <p className="composer__hint">
          <kbd>Enter</kbd> to send · <kbd>Shift</kbd>+<kbd>Enter</kbd> for a new line ·
          answers come only from your documents
        </p>
      </div>
    </div>
  );
}
