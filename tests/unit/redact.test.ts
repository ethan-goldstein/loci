import { describe, expect, it } from 'vitest';
import { entropy, redact } from '../../src/ingest/redact.ts';
import { rng } from '../../src/prove/synth.ts';

const ALNUM = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
const HEX = '0123456789abcdef';

function str(rand: () => number, alphabet: string, len: number): string {
  let s = '';
  for (let i = 0; i < len; i++) s += alphabet[Math.floor(rand() * alphabet.length)];
  return s;
}

/** One generator per secret family. Each returns [textContainingSecret, theSecret]. */
const FAMILIES: Record<string, (r: () => number) => [string, string]> = {
  anthropic: (r) => {
    const s = `sk-ant-api03-${str(r, ALNUM, 48)}`;
    return [`export ANTHROPIC=${s}`, s];
  },
  openai: (r) => {
    const s = `sk-proj-${str(r, ALNUM, 40)}`;
    return [`the key is ${s} ok`, s];
  },
  github: (r) => {
    const s = `ghp_${str(r, ALNUM, 36)}`;
    return [`git remote set-url origin https://x@github.com with ${s}`, s];
  },
  githubPat: (r) => {
    const s = `github_pat_${str(r, ALNUM, 60)}`;
    return [`token: ${s}`, s];
  },
  aws: (r) => {
    const s = `AKIA${str(r, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', 16)}`;
    return [`aws id ${s} in config`, s];
  },
  google: (r) => {
    const s = `AIza${str(r, ALNUM, 35)}`;
    return [`maps key ${s}`, s];
  },
  slack: (r) => {
    const s = `xoxb-${str(r, '0123456789', 12)}-${str(r, ALNUM, 24)}`;
    return [`SLACK ${s}`, s];
  },
  stripe: (r) => {
    const s = `sk_live_${str(r, ALNUM, 24)}`;
    return [`stripe.setKey("${s}")`, s];
  },
  shopify: (r) => {
    const s = `shpat_${str(r, HEX, 32)}`;
    return [`X-Shopify-Access-Token ${s}`, s];
  },
  jwt: (r) => {
    const s = `eyJ${str(r, ALNUM, 20)}.eyJ${str(r, ALNUM, 40)}.${str(r, ALNUM, 30)}`;
    return [`cookie session=${s};`, s];
  },
  pem: (r) => {
    const body = str(r, ALNUM, 200);
    const s = `-----BEGIN RSA PRIVATE KEY-----\n${body}\n-----END RSA PRIVATE KEY-----`;
    return [`here is my key\n${s}\nthanks`, body];
  },
  urlCredentials: (r) => {
    const pw = str(r, ALNUM, 14);
    return [`DATABASE at postgres://admin:${pw}@db.internal:5432/app`, pw];
  },
  bearer: (r) => {
    const s = str(r, ALNUM, 40);
    return [`curl -H "Authorization: Bearer ${s}" https://api.example.com`, s];
  },
  envAssignment: (r) => {
    const s = str(r, 'abcdefghijklmnopqrstuvwxyz', 12); // low entropy: only the name gives it away
    return [`SMTP_PASSWORD=${s}\nPORT=587`, s];
  },
  jsonSecret: (r) => {
    const s = str(r, HEX, 40); // hex: the entropy rule ignores it, the name rule must catch it
    return [`{"client_secret": "${s}", "port": 3000}`, s];
  },
  yamlToken: (r) => {
    const s = str(r, ALNUM, 18);
    return [`auth_token: '${s}'`, s];
  },
  highEntropy: (r) => {
    const s = `${str(r, ALNUM, 44)}==`;
    return [`paste this blob ${s} into the field`, s];
  },
  email: (r) => {
    const s = `${str(r, 'abcdefghijklmnopqrstuvwxyz', 8)}@example-mail.com`;
    return [`send it to ${s} please`, s];
  },
  phone: (r) => {
    const s = `(${str(r, '23456789', 3)}) ${str(r, '0123456789', 3)}-${str(r, '0123456789', 4)}`;
    return [`call me at ${s} tomorrow`, s];
  },
  homePath: (r) => {
    const user = `user${str(r, 'abcdefghijklmnopqrstuvwxyz', 8)}`;
    return [`open /Users/${user}/projects/app/main.ts`, user];
  },
};

describe('redact', () => {
  it('removes every generated secret, across 200 seeds per family', () => {
    for (const [family, make] of Object.entries(FAMILIES)) {
      for (let seed = 1; seed <= 200; seed++) {
        const [text, secret] = make(rng(seed * 7919 + family.length));
        const out = redact(text);
        expect(out.text.includes(secret), `${family} seed ${seed} leaked`).toBe(false);
        expect(Object.values(out.counts).reduce((a, b) => a + b, 0), `${family} seed ${seed} not counted`).toBeGreaterThan(0);
      }
    }
  });

  it('survives secrets glued to surrounding punctuation', () => {
    const r = rng(42);
    const key = `ghp_${str(r, ALNUM, 36)}`;
    for (const wrap of [`"${key}"`, `(${key})`, `token=${key},`, `<${key}>`, `\`${key}\``, `${key}\n`]) {
      expect(redact(wrap).text.includes(key)).toBe(false);
    }
  });

  it('leaves ordinary code and prose alone', () => {
    const keep = [
      'const apiKey = process.env.API_KEY;',
      'commit 3f2a9c1b7d4e5f60718293a4b5c6d7e8f9a0b1c2 fixed the parser',
      'npm install three@0.186.0 vite@8.3.0',
      'the function returnsTheLongestCommonSubsequenceOfTwoStrings is slow',
      'password must be at least 12 characters',
      'run `git status` then open src/index/hnsw.ts:120',
      'set TOKEN=<your-token-here> in .env',
      'tolerance is 0.0001 and the count was 123-456',
    ];
    for (const text of keep) expect(redact(text).text).toBe(text);
  });

  it('keeps the name of a named secret so the text stays readable', () => {
    const out = redact('SMTP_PASSWORD=hunter2hunter2');
    expect(out.text).toBe('SMTP_PASSWORD=[REDACTED:named-secret]');
    expect(out.counts['named-secret']).toBe(1);
  });

  it('rewrites home directories to ~', () => {
    expect(redact('/Users/somebody/code/x.ts and /home/other/y').text).toBe('~/code/x.ts and ~/y');
    expect(redact('-Users-somebody-code-app').text).toBe('-~-code-app');
  });

  it('redacts partly masked addresses but not a bare domain mention', () => {
    expect(redact('sent to (ab…@example-mail.com) today').text).toBe('sent to ([REDACTED:email]) today');
    expect(redact('login j***@example-mail.com ok').text).toBe('login [REDACTED:email] ok');
    expect(redact('anyone with an @example-mail.com address').text).toBe('anyone with an @example-mail.com address');
  });

  it('is idempotent', () => {
    const [text] = FAMILIES.stripe!(rng(5));
    const once = redact(`${text} and mail a@b.co`).text;
    expect(redact(once).text).toBe(once);
  });

  it('computes entropy sensibly', () => {
    expect(entropy('aaaaaaaa')).toBe(0);
    expect(entropy('abcdefgh')).toBeCloseTo(3, 5);
  });
});
