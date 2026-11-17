/**
 * Secret and personal-data redaction. Runs before any text is hashed, stored, embedded or served,
 * so nothing downstream ever sees the original value.
 *
 * This is a best-effort filter, not a guarantee: it recognises known credential formats, values
 * attached to secret-looking names, and high-entropy runs. A secret that looks like an ordinary
 * word will pass. The fuzz test in tests/unit/redact.test.ts pins what is covered.
 */

export type RedactionCounts = Record<string, number>;

export interface RedactionResult {
  text: string;
  counts: RedactionCounts;
}

interface Rule {
  family: string;
  re: RegExp;
  /** Replacement; defaults to `[REDACTED:<family>]`. */
  replace?: (match: string, ...groups: string[]) => string;
}

const mark = (family: string) => `[REDACTED:${family}]`;

/** Vendor credential formats with a fixed, recognisable prefix. */
const VENDOR_KEYS: RegExp[] = [
  /\bsk-ant-[A-Za-z0-9_-]{20,}/g, // Anthropic
  /\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{32,}/g, // OpenAI-style
  /\bgh[pousr]_[A-Za-z0-9]{30,}/g, // GitHub tokens
  /\bgithub_pat_[A-Za-z0-9_]{40,}/g,
  /\b(?:AKIA|ASIA)[0-9A-Z]{16}\b/g, // AWS access key id
  /\bAIza[0-9A-Za-z_-]{35}\b/g, // Google API key
  /\bya29\.[0-9A-Za-z_-]{20,}/g, // Google OAuth access token
  /\bxox[abeoprs]-[A-Za-z0-9-]{10,}/g, // Slack
  /\b(?:sk|pk|rk)_(?:live|test)_[A-Za-z0-9]{16,}/g, // Stripe
  /\bwhsec_[A-Za-z0-9]{24,}/g, // Stripe webhook secret
  /\bshp(?:at|ca|pa|ss)_[a-fA-F0-9]{32}\b/g, // Shopify
  /\bhf_[A-Za-z0-9]{30,}/g, // Hugging Face
  /\bnpm_[A-Za-z0-9]{36}\b/g,
  /\bSG\.[A-Za-z0-9_-]{16,}\.[A-Za-z0-9_-]{16,}/g, // SendGrid
  /\bglpat-[A-Za-z0-9_-]{20,}/g, // GitLab
  /\bdop_v1_[a-f0-9]{64}\b/g, // DigitalOcean
  /\btskey-[A-Za-z0-9-]{20,}/g, // Tailscale
  /\bAC[a-f0-9]{32}\b/g, // Twilio account SID
  /\bSK[a-f0-9]{32}\b/g, // Twilio API key
];

const SECRET_NAME =
  '(?:[A-Za-z0-9_.-]*(?:passwords?|passwd|pwd|secrets?|tokens?|api[_-]?keys?|apikeys?|access[_-]?keys?|private[_-]?keys?|client[_-]?secrets?|credentials?|auth(?:orization)?|signing[_-]?keys?|hmac(?:[_-]?keys?)?|session[_-]?keys?)[A-Za-z0-9_.-]*)';

const RULES: Rule[] = [
  {
    family: 'private-key',
    re: /-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----[\s\S]*?(?:-----END [A-Z0-9 ]*PRIVATE KEY-----|$)/g,
  },
  { family: 'jwt', re: /\beyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{6,}/g },
  ...VENDOR_KEYS.map((re) => ({ family: 'vendor-key', re })),
  {
    family: 'url-credentials',
    re: /\b([a-z][a-z0-9+.-]{1,20}:\/\/)[^\s/:@]{1,128}:[^\s/@]{1,256}@/gi,
    replace: (_m, scheme) => `${scheme}${mark('url-credentials')}@`,
  },
  {
    family: 'bearer',
    re: /\b(Bearer|Basic)\s+[A-Za-z0-9._~+/=-]{16,}/g,
    replace: (_m, kind) => `${kind} ${mark('bearer')}`,
  },
  {
    // NAME=value, NAME: value, "name": "value". Keeps the name, drops the value.
    family: 'named-secret',
    re: new RegExp(
      `(["'\`]?\\b${SECRET_NAME}["'\`]?[ \\t]*[:=][ \\t]*)(["'\`]?)([^\\s"'\`,;)}\\]]{6,})\\2`,
      'gi',
    ),
    replace: (m, lead, _q, value) => {
      // Leave obvious non-secrets alone so code stays readable.
      if (/^(?:process\.env|import\.meta|os\.environ|env\.|\$\{?[A-Za-z_]|<[^>]*>?$|true$|false$|null$|undefined$|none$|string$|required$)/i.test(value)) {
        return m;
      }
      if (value.startsWith('[REDACTED:')) return m;
      return `${lead}${mark('named-secret')}`;
    },
  },
];

// The local part may already be partly masked ("ab…@example.com", "a***@example.com"); those are still addresses.
const EMAIL = /(?<![A-Za-z0-9._%+*…-])[A-Za-z0-9._%+*…-]{1,64}@[A-Za-z0-9-]{1,63}(?:\.[A-Za-z0-9-]{1,63})*\.[A-Za-z]{2,24}\b/g;
const PHONE = /(?<![\w.-])(?:\+?1[\s.-]?)?(?:\(\d{3}\)\s?|\d{3}[\s.-])\d{3}[\s.-]\d{4}(?![\w-])/g;
const HOME_PATH = /(?:\/Users|\/home)\/[A-Za-z0-9._-]+/g;
const ENCODED_HOME = /-(?:Users|home)-[A-Za-z0-9._]+(?=-)/g;
const LONG_TOKEN = /[A-Za-z0-9+/=_-]{32,}/g;

/** Shannon entropy in bits per character. */
export function entropy(s: string): number {
  const freq = new Map<string, number>();
  for (const ch of s) freq.set(ch, (freq.get(ch) ?? 0) + 1);
  let h = 0;
  for (const n of freq.values()) {
    const p = n / s.length;
    h -= p * Math.log2(p);
  }
  return h;
}

/**
 * A long unbroken run that looks random: mixed character classes and high entropy. Hex-only
 * strings (git SHAs, content hashes) max out at 4 bits/char and are left alone; secrets in hex are
 * caught by the named-secret rule when they sit next to a telling name.
 */
function looksRandom(tok: string): boolean {
  if (/^[0-9a-fA-F]+$/.test(tok)) return false;
  if (/^[A-Za-z]+$/.test(tok) || /^[0-9]+$/.test(tok)) return false;
  if (/^[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+){2,}$/.test(tok) && !/\d/.test(tok)) return false; // kebab/snake identifiers
  const hasLower = /[a-z]/.test(tok);
  const hasUpper = /[A-Z]/.test(tok);
  const hasDigit = /[0-9]/.test(tok);
  if (Number(hasLower) + Number(hasUpper) + Number(hasDigit) < 3) return false;
  return entropy(tok) >= 4.3;
}

function bump(counts: RedactionCounts, family: string): void {
  counts[family] = (counts[family] ?? 0) + 1;
}

export function redact(input: string): RedactionResult {
  const counts: RedactionCounts = {};
  let text = input;

  for (const rule of RULES) {
    text = text.replace(rule.re, (m: string, ...groups: string[]) => {
      const out = rule.replace ? rule.replace(m, ...groups) : mark(rule.family);
      if (out !== m) bump(counts, rule.family);
      return out;
    });
  }

  text = text.replace(LONG_TOKEN, (tok) => {
    if (!looksRandom(tok)) return tok;
    bump(counts, 'high-entropy');
    return mark('high-entropy');
  });

  text = text.replace(EMAIL, () => {
    bump(counts, 'email');
    return mark('email');
  });

  text = text.replace(PHONE, () => {
    bump(counts, 'phone');
    return mark('phone');
  });

  text = text.replace(HOME_PATH, () => {
    bump(counts, 'home-path');
    return '~';
  });
  text = text.replace(ENCODED_HOME, () => {
    bump(counts, 'home-path');
    return '-~';
  });

  return { text, counts };
}

export function mergeCounts(into: RedactionCounts, from: RedactionCounts): void {
  for (const [k, v] of Object.entries(from)) into[k] = (into[k] ?? 0) + v;
}
