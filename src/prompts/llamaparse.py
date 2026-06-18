LLAMAPARSE_PROMPT = """

This is a SEC Form 10-K or 10-Q filing. STRICT OUTPUT RULES:

HEADING HIERARCHY:
- Only 'PART I', 'PART II', 'PART III', 'PART IV' use heading level # (H1). Nothing else uses #.
- Only 'Item X.' entries (Item 1., 1A., 1B., 1C., 2., 3., 4., 5., 6., 7., 7A., 8., 9., 9A., 9B., 9C., 10., 11., 12., 13., 14., 15., 16.) use heading level ## (H2). Nothing else uses ##.
- All headings based on visual hierarchy, text size, and style should be of ### (H3). 
- Hierarchy resets per Item.

FORMATTING RULES:
- MUST parse TABLE perfectly 100 percent exact same as its in the page .
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