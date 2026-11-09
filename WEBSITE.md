# Website

The buildless website is in `docs/`, following the structure of the comparative atlas. It works without JavaScript and uses native MathML for equations.

## Preview

From the repository root:

```bash
python -m http.server 8000 --directory docs
```

Open http://localhost:8000/.

## Publish on GitHub Pages

1. Upload this public repository folder to the intended GitHub repository. Do not upload the parent manuscript working directory.
2. In **Settings → Pages**, choose **Deploy from a branch**, then branch **main** and folder **/docs**.
3. Save. GitHub will show the website URL after deployment.

All site links are relative, so no repository name needs replacing before deployment. After publication, add the actual website and repository links to the README and citation metadata. Add the canonical site URL and absolute `citation_pdf_url` metadata to the HTML pages for scholarly indexing.

## Archival record

Zenodo: https://zenodo.org/records/24084198

DOI: https://doi.org/10.5281/zenodo.24084198

For a future manuscript version, update the citation metadata and website links together.

## Files and maintenance

- `index.html`: overview and abstract.
- `full-text.html`: HTML manuscript, generated privately from the LaTeX source.
- `gallery.html`: all nine manuscript figures and their original captions.
- `videos.html`: both videos, with native playback controls.
- `resources.html`: code download and Zenodo citation.
- `manuscript.pdf`: reference paper.
- `assets/`: optimised website images and videos.
- `downloads/`: original figure PDFs and a ZIP of the scripts.

The public repository deliberately contains no LaTeX source. Replace the two manuscript PDFs together when publishing a new version. Regenerate the HTML and gallery in the private working folder before exporting another release. The private `scripts/export_public.py` helper performs this export and is not part of the public distribution.
