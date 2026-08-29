/** One chat turn, with expandable source citations. */

import { useState } from 'react';
import type { Message } from '../lib/types';
import { renderMarkdown } from '../lib/format';
import { IconAlert, IconChevron, IconCopy, IconQuote, IconRefresh } from './Icons';

interface ChatMessageProps {
  message: Message;
  canRegenerate: boolean;
  onRegenerate: () => void;
}

export function ChatMessage({
  message,
  canRegenerate,
  onRegenerate,
}: ChatMessageProps) {
  const [showSources, setShowSources] = useState(false);
  const [copied, setCopied] = useState(false);

  const isUser = message.role === 'user';
  const sources = message.sources ?? [];
  const ungrounded = message.grounded === false && !message.error;

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(message.content);
      setCopied(true);
      setTimeout(() => setCopied(false), 1600);
    } catch {
      // Clipboard access can be denied; failing silently is acceptable here.
    }
  };

  return (
    <article className="msg" data-role={message.role}>
      <div className="msg__avatar" aria-hidden="true">
        {isUser ? 'You' : 'AI'}
      </div>

      <div className="msg__body">
        <div
          className="msg__bubble"
          data-error={message.error || undefined}
          data-ungrounded={ungrounded || undefined}
        >
          {isUser ? (
            <span style={{ whiteSpace: 'pre-wrap' }}>{message.content}</span>
          ) : message.content ? (
            <>
              <span
                className="md"
                // renderMarkdown escapes all HTML before inserting tags.
                dangerouslySetInnerHTML={{ __html: renderMarkdown(message.content) }}
              />
              {message.streaming && <span className="caret" />}
            </>
          ) : (
            <div className="typing" aria-label="Assistant is thinking">
              <span />
              <span />
              <span />
            </div>
          )}
        </div>

        {/* ---------- Citations ---------- */}
        {!isUser && sources.length > 0 && (
          <div className="sources">
            <button
              type="button"
              className="sources__toggle"
              data-open={showSources}
              aria-expanded={showSources}
              onClick={() => setShowSources((value) => !value)}
            >
              <IconQuote size={12} />
              {sources.length} source{sources.length === 1 ? '' : 's'}
              <IconChevron size={12} />
            </button>

            {showSources && (
              <div className="sources__list">
                {sources.map((source) => (
                  <div className="source" key={source.chunk_id}>
                    <div className="source__head">
                      <span className="source__label">{source.label}</span>
                      {source.section && source.page !== null && (
                        <span className="badge">{source.section}</span>
                      )}
                      <span className="source__score">
                        {(source.score * 100).toFixed(0)}% match
                      </span>
                    </div>
                    <p className="source__excerpt">{source.excerpt}</p>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* ---------- Actions ---------- */}
        {!isUser && !message.streaming && message.content && (
          <div className="msg__footer">
            <button type="button" className="msg__action" onClick={copy}>
              <IconCopy />
              {copied ? 'Copied' : 'Copy'}
            </button>
            {canRegenerate && (
              <button
                type="button"
                className="msg__action"
                onClick={onRegenerate}
                title="Ask the same question again"
              >
                <IconRefresh />
                Regenerate
              </button>
            )}
            {ungrounded && (
              <span
                className="msg__action"
                style={{ color: 'var(--warning)', cursor: 'default' }}
                title="The answer was not found in the selected documents."
              >
                <IconAlert size={12} />
                Not in document
              </span>
            )}
          </div>
        )}
      </div>
    </article>
  );
}
