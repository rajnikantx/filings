v0 = """

This is a SEC Form 10-K or 10-Q filing. STRICT OUTPUT RULES:

HEADING HIERARCHY:
- Only 'PART I', 'PART II', 'PART III', 'PART IV' use heading level # (H1). Nothing else uses #.
- Only 'Item X.' entries (Item 1., 1A., 1B., 1C., 2., 3., 4., 5., 6., 7., 7A., 8., 9., 9A., 9B., 9C., 10., 11., 12., 13., 14., 15., 16.) use heading level ## (H2). Nothing else uses ##.
- All headings based on visual hierarchy, text size, and style should be of ### (H3). 
- Hierarchy resets per Item.

FORMATTING RULES:
- NEVER USE any headings except H1, H2, H3.
- NEVER use **bold**, *italic*, __underline__, `code`, or any text formatting
- NEVER use HTML tags like <b>, <i>, <strong>, <em>
- NEVER use markdown * emphasis characters
- NEVER use markdown emphasis characters: *, _, `, ~, >, -, +, |
- NEVER use bullet points, numbered lists, tables, blockquotes, or horizontal rules
- Output headings as plain text with # symbols only
- All body text must be plain text without any formatting wrappers
- Preserve exact Item numbering and titles
- Never skip heading levels
- Never use # or ## inside Item content

"""




v1 = """

This is a SEC Form 10-K filing. STRICT OUTPUT RULES:

HEADING HIERARCHY:
- Only 'PART I', 'PART II', 'PART III', 'PART IV' use heading level # (H1). Nothing else uses #.
- Only 'Item X.' entries (Item 1., 1A., 1B., 1C., 2., 3., 4., 5., 6., 7., 7A., 8., 9., 9A., 9B., 9C., 10., 11., 12., 13., 14., 15., 16.) use heading level ## (H2). Nothing else uses ##.
- All content under each Item starts from ### (H3) onwards through ###### (H6), based on visual hierarchy, text size, and style. Hierarchy resets per Item.

FORMATTING RULES:
- NEVER use **bold**, *italic*, __underline__, `code`, or any text formatting
- NEVER use HTML tags like <b>, <i>, <strong>, <em>
- NEVER use markdown * emphasis characters
- NEVER use markdown emphasis characters: *, _, `, ~, >, -, +, |
- NEVER use bullet points, numbered lists, tables, blockquotes, or horizontal rules
- Output headings as plain text with # symbols only
- All body text must be plain text without any formatting wrappers
- Preserve exact Item numbering and titles
- Never skip heading levels
- Never use # or ## inside Item content

"""


v2 = """

This is a SEC Form 10-K filing. STRICT OUTPUT RULES:

HEADING HIERARCHY:
- ALL headings in the document must use ONLY heading level # (H1). 
- No other heading levels allowed (##, ###, ####, #####, ###### are FORBIDDEN).
- Any text that is visually distinct (larger font, more whitespace above/below, bold, centered, title case, short phrase followed by long paragraph) must be marked as # heading.
- Body text (smaller, regular weight, continuous paragraphs) must NOT have any heading marker.

SPECIAL RULES:
- The very first heading must be "## Intro" for all pre-PART I content.
- After that, ALL headings (PART I, Item 1., subsections, etc.) use ONLY #.
- Do not use multiple heading levels. Flatten everything to #.

FORMATTING:
- NEVER use **bold**, *italic*, __underline__, `code`, or any text formatting
- NEVER use markdown emphasis characters: *, _, `, ~, >, -, +, |
- NEVER use bullet points, numbered lists, tables, blockquotes
- Output headings as plain text with # symbol only
- All body text must be plain text without any formatting wrappers
- Preserve exact text content

"""


v3 = """

DOCUMENT PARSING ENGINE - VISUAL HIERARCHY MODE

MISSION: Detect headings by comparing text size, weight, and spacing. NO EXCEPTIONS.

DETECTION LOGIC:
For each text block, ask:
- Is this LARGER than the text below it? → HEADING
- Is this BOLDER than the text below it? → HEADING  
- Does this have MORE space above it? → HEADING
- Is this SHORTER than the text below it? → HEADING

If YES to any → assign heading level based on relative size
If NO to all → body text

HEADING LEVELS:
Largest on page = #
Next largest = ##
Next = ###
Next = ####
Next = #####
Next = ######
Everything else = body text

OUTPUT:
- Only # to ###### for headings
- NO **bold**, *italic*, __underline__, `code`
- NO *, _, `, ~, >, -, +, |
- NO lists, tables, blockquotes
- Plain text only

"""


v4 = """

You are a document parsing engine. Your task is to extract text and assign heading levels based STRICTLY on visual formatting cues.

HEADING DETECTION RULES (Apply to ALL documents):
1. The LARGEST text on the page = heading level # (H1)
2. The NEXT largest text = heading level ## (H2)
3. Continue down: ### (H3), #### (H4), ##### (H5), ###### (H6)
4. The SMALLEST text = body text (no heading)
5. Text with MORE whitespace above/below = higher heading level
6. Text with DIFFERENT font weight (bold vs regular) = different heading level
7. Short phrases (2-8 words) followed by long paragraphs = heading
8. Title case text followed by sentence case text = heading

CRITICAL: Compare EVERY text block to its neighbors. If one block is visually distinct (larger, bolder, more spacing), it MUST be a heading. If two blocks look similar, they are the SAME level.

OUTPUT RULES:
- Only use # to ###### for headings
- Never use **bold**, *italic*, __underline__, `code`, or any text formatting
- Never use markdown emphasis characters: *, _, `, ~, >, -, +, |
- Never use bullet points, numbered lists, tables, blockquotes, or horizontal rules
- All body text must be plain text without any formatting wrappers
- Never skip heading levels
- Output headings as plain text with # symbols only

"""

