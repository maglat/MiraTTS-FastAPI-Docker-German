import re
import logging

logger = logging.getLogger(__name__)


class TextProcessor:
    def __init__(self, max_chars: int = 200):
        self.max_chars = max_chars

    def _split_into_sentences(self, text: str):
        """
        German-friendly sentence splitter.

        Key change vs. previous version:
        - Removed the English-style heuristic that detects sentence boundaries via:
          lowercase + space + Uppercase
          This breaks German badly because nouns are capitalized mid-sentence.

        We primarily split on sentence end punctuation: . ! ?
        We also support quotes/brackets after punctuation, protect abbreviations and decimals.
        """
        if not text or not text.strip():
            return []

        # Normalize whitespace (keep meaning, just reduce noise)
        text = re.sub(r"\s+", " ", text.strip())

        # --- Abbreviation protection -------------------------------------------------
        # 1) Explicit list (kept + extended for German)
        abbreviations = {
            # EN / generic (kept from your file)
            "Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Sr.", "Jr.",
            "vs.", "etc.", "i.e.", "e.g.", "cf.", "al.", "Co.",
            "Corp.", "Inc.", "Ltd.", "St.", "Ave.", "Blvd.",
            "U.S.", "U.K.", "U.N.", "E.U.", "Ph.D.", "M.D.",
            "B.A.", "M.A.", "a.m.", "p.m.", "A.M.", "P.M.",

            # DE common abbreviations
            "z.B.", "d.h.", "u.a.", "u.U.", "u.Ä.", "u.ä.", "u.v.m.", "u.s.w.", "usw.",
            "bzw.", "ca.", "vgl.", "evtl.", "ggf.", "Nr.", "Art.", "Abs.", "Kap.", "Abb.",
            "S.", "Hr.", "Fr.", "Dr.", "Prof.", "Dipl.", "Ing.", "inkl.", "zzgl.",
            "MwSt.", "o.Ä.", "o.ä.", "s.o.", "s.u.",
        }

        # 2) Regex-based patterns for German abbreviations (more robust than exact replace)
        #    - single-letter initials like "A." inside names should generally NOT end sentence
        #    - multi-dot abbreviations like "z.B." / "d.h." / "u.a."
        # We will replace matches with placeholders.
        protected_tokens = {}

        def _protect(pattern: str, label: str):
            nonlocal text
            idx = 0

            def repl(m):
                nonlocal idx
                token = m.group(0)
                key = f"__{label}_{idx}__"
                protected_tokens[key] = token
                idx += 1
                return key

            text = re.sub(pattern, repl, text)

        # Protect explicit abbreviations first (fast path)
        for i, abbrev in enumerate(sorted(abbreviations, key=len, reverse=True)):
            if abbrev in text:
                key = f"__ABBREV_{i}__"
                text = text.replace(abbrev, key)
                protected_tokens[key] = abbrev

        # Protect decimals like 3.14
        decimal_pattern = r"\b\d+\.\d+\b"
        _protect(decimal_pattern, "DECIMAL")

        # Protect multi-dot abbrevs like "z.B." "d.h." "u.a." even if not in explicit list
        # (letters with dots, 2-4 segments)
        _protect(r"\b(?:[A-Za-zÄÖÜäöüß]\.){2,4}\b", "MULTIDOT")

        # Protect German constructs like "z. B." / "d. h." with spaces
        _protect(r"\b(?:[A-Za-zÄÖÜäöüß]\.\s*){2,4}\b", "MULTIDOT_SP")

        # Protect single-letter initials like "A." when followed by uppercase letter (surname)
        _protect(r"\b[A-ZÄÖÜ]\.(?=\s+[A-ZÄÖÜ])", "INITIAL")

        # --- Sentence splitting -------------------------------------------------------
        # In German, ":" and ";" are often mid-sentence. Keep off by default.
        treat_colon_semicolon_as_end = False
        end_punct = {".", "!", "?"}
        if treat_colon_semicolon_as_end:
            end_punct |= {":", ";"}

        sentences = []
        current = ""
        i = 0

        # Characters that may follow punctuation but should be kept with the sentence end
        trailing_closers = set(['"', "'", "”", "“", "’", "»", "«", ")", "]", "}", "›", "‹"])

        while i < len(text):
            ch = text[i]
            current += ch

            if ch in end_punct:
                # absorb trailing quotes/brackets right after punctuation
                j = i + 1
                while j < len(text) and text[j] in trailing_closers:
                    current += text[j]
                    j += 1

                # Decide if this is a real sentence end.
                # We require that the next non-space char (if any) is not lowercase continuation.
                # (This helps with things like: "Hallo! und dann..." where user forgot uppercase;
                #  but German typically starts new sentence with uppercase; still we avoid hard dependence.)
                remaining = text[j:].lstrip()
                is_sentence_end = True

                if remaining:
                    next_char = remaining[0]

                    # If next char is lowercase, it might be a continuation (e.g. ellipsis or stylistic)
                    # We'll be conservative: do NOT split if punctuation is '.' and there is only one space
                    # and the next begins with lowercase.
                    if next_char.islower() and ch == ".":
                        # check whitespace length between i and next non-space
                        gap = len(text[j:]) - len(remaining)
                        if gap <= 1:
                            is_sentence_end = False

                    # Also avoid splitting after ALLCAP short tokens (kept from your logic)
                    # Example: "USA. ist ..." might or might not be split; we keep your safeguard.
                    if ch == "." and current:
                        words_before = current.strip().split()
                        if words_before:
                            last = words_before[-1]
                            if len(last) <= 4 and last.isupper():
                                # likely an acronym; might not be sentence end
                                # but if we see two spaces or end of text, we allow split
                                gap = len(text[j:]) - len(remaining)
                                if gap <= 1:
                                    is_sentence_end = False

                if is_sentence_end:
                    sentence = current.strip()
                    if sentence:
                        sentences.append(sentence)
                    current = ""
                    i = j
                    continue

            i += 1

        if current.strip():
            sentences.append(current.strip())

        # Restore protected tokens (abbrevs, decimals, etc.)
        restored = []
        for s in sentences:
            for placeholder, original in protected_tokens.items():
                s = s.replace(placeholder, original)
            restored.append(s)

        # Final cleanup / merge tiny fragments (kept from your file with small tweak)
        clean_sentences = []
        for sentence in restored:
            sentence = sentence.strip()
            if not sentence:
                continue

            # If it's a very short fragment without final punctuation, merge to previous
            if (
                len(clean_sentences) > 0
                and len(sentence.split()) <= 3
                and (not sentence.endswith((".", "!", "?")))
                and not sentence.startswith('"')
            ):
                clean_sentences[-1] += " " + sentence
            else:
                clean_sentences.append(sentence)

        return clean_sentences

    def chunk_text(self, text: str) -> list[str]:
        """
        Packs WHOLE sentences into chunks.
        It will only break a sentence if that single sentence exceeds max_chars.

        German tweak:
        - If a single sentence is too long, prefer splitting on commas/;/:/dashes before plain spaces.
        """
        sentences = self._split_into_sentences(text)
        chunks = []
        current_chunk = ""

        for s in sentences:
            # If the chunk + this sentence is still within limits, combine them
            if len(current_chunk) + len(s) + (1 if current_chunk else 0) <= self.max_chars:
                current_chunk = (current_chunk + " " + s).strip()
                continue

            # Save the current chunk if it has content
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""

            # If this single sentence is still longer than max_chars, we must split it
            if len(s) > self.max_chars:
                # Prefer "natural" breakpoints first for German:
                # comma, semicolon, colon, em/en-dash; then fallback to spaces.
                # Keep punctuation with the left part (lookbehind).
                sub_parts = re.split(r"(?<=[,;:–—-])\s+|\s+", s)

                temp = ""
                for part in sub_parts:
                    if not part:
                        continue
                    if len(temp) + len(part) + (1 if temp else 0) > self.max_chars:
                        if temp:
                            chunks.append(temp.strip())
                        temp = part
                    else:
                        temp = (temp + " " + part).strip()

                if temp:
                    current_chunk = temp
            else:
                # Sentence fits in a new empty chunk
                current_chunk = s

        if current_chunk:
            chunks.append(current_chunk)

        return chunks
