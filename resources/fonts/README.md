# Fonts of the Supplementary Note

`study/report/render.py` inlines these files as base64 `@font-face` rules in `supplementary_note.html` and `supplementary_note.print.html`, so the Note shows, and Chrome prints, the fonts report v10 loaded from Google Fonts without fetching anything. Each file is copied unchanged from the official release named below (WOFF2, not subset).

| File | Family and weight in the Note | Release |
|---|---|---|
| `IBMPlexSans-Regular.woff2`, `IBMPlexSans-Medium.woff2`, `IBMPlexSans-SemiBold.woff2` | IBM Plex Sans 400, 500, 600 | IBM Plex Sans 1.1.0, GitHub release `@ibm/plex-sans@1.1.0` of github.com/IBM/plex (2024-11-13), `ibm-plex-sans.zip`, `fonts/complete/woff2/` |
| `IBMPlexMono-Regular.woff2`, `IBMPlexMono-Medium.woff2` | IBM Plex Mono 400, 500 | IBM Plex Mono 2.5.0, GitHub release `@ibm/plex-mono@2.5.0` of github.com/IBM/plex (2026-06-11), `ibm-plex-mono.zip`, `fonts/complete/woff2/` |
| `SourceSerif4SmText-Regular.ttf.woff2`, `SourceSerif4SmText-Semibold.ttf.woff2`, `SourceSerif4SmText-It.ttf.woff2` | Source Serif 4 400, 600 and 400 italic: the text | Source Serif 4.005, GitHub release `4.005R` of github.com/adobe-fonts/source-serif (2023-01-20), `source-serif-4.005_WOFF2.zip`, `TTF/` |
| `SourceSerif4Subhead-Regular.ttf.woff2`, `SourceSerif4Subhead-Semibold.ttf.woff2` | Source Serif 4 Subhead 400, 600: the title | the same Source Serif release |

v10 loaded Source Serif 4 as a variable font with an optical-size axis, so the browser set the optical size from the font size: 15.5 to 17 px for the text, 27 to 34 px for the title. The Note uses the release's static cuts for those sizes instead: Small Text (optical sizes 12 to 18) and Subhead (26 to 48). Chrome writes a variable font into a PDF as unnamed Type 3 glyphs; static cuts are embedded by name.

Licenses: SIL Open Font License 1.1. `OFL-IBM-Plex.txt` is `LICENSE.txt` of both IBM Plex releases (the same text in each); `OFL-Source-Serif.md` is `LICENSE.md` of the Source Serif repository at tag `4.005R`.
