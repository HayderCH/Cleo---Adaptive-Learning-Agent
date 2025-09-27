Mermaid diagrams

Preview options

- VS Code: install the “Markdown Preview Mermaid Support” extension and open any .md that embeds a ```mermaid block, or open .mmd files with a Mermaid preview extension.
- Browser: use mermaid.live to paste the contents of any .mmd file.
- CLI export: install Node + @mermaid-js/mermaid-cli and export to PNG/SVG.

CLI export (Windows PowerShell)

```powershell
# Install once
npm install -g @mermaid-js/mermaid-cli

# Export examples
mmdc -i .\docs\diagrams\interaction-sequence.mmd -o .\docs\diagrams\interaction-sequence.png
mmdc -i .\docs\diagrams\component-map.mmd -o .\docs\diagrams\component-map.png
mmdc -i .\docs\diagrams\schema-sample.mmd -o .\docs\diagrams\schema-sample.png
mmdc -i .\docs\diagrams\coach-policy-timeline.mmd -o .\docs\diagrams\coach-policy-timeline.png
mmdc -i .\docs\diagrams\spaced-review-gantt.mmd -o .\docs\diagrams\spaced-review-gantt.png
```

Files

- interaction-sequence.mmd — end-to-end sequence
- component-map.mmd — mindmap view of components
- schema-sample.mmd — Sample schema mindmap
- coach-policy-timeline.mmd — coach bands as a timeline
- spaced-review-gantt.mmd — spaced repetition schedule example
