# OALD9 Lingea Extractor

`extract_oald9.py` extracts usable assets from the Oxford Advanced Learner's Dictionary 9th Edition data files used by the Lingea container format.

It can extract:

- dictionary article records from `oup_en-dic.lld` into decoded XML-like text
- images from `oup_en-dic.mbx`
- audio from `oup_en-dic.wbx`
- optional diagnostic views for reverse engineering the binary dictionary records

The decoded dictionary output is not plain HTML. It is Lingea/Oxford internal article markup, with tags such as `<c.entry>`, `<c.h>`, `<c.def>`, `<c.pron-g>`, and audio references such as `abandon#_gb_1`, which requires custom parsing implementation.

## Prerequisites

- Python 3.10 or newer
- The original `data` directory beside the script, or passed with `--data-dir`
- These source files in the data directory:
  - `oup_en-dic.lld`
  - `oup_en-dic.mbx`
  - `oup_en-dic.wbx`

## Quick Start

```powershell
python extract_oald9.py
```

With no flags, the script extracts media and dictionary records into `extracted`.

## Common Commands

Extract only decoded dictionary XML:

```powershell
python extract_oald9.py --dictionary --out-dir extracted_dictionary
```

Extract only images:

```powershell
python extract_oald9.py --images --out-dir extracted_images
```

Extract only audio:

```powershell
python extract_oald9.py --audio --out-dir extracted_audio
```

Extract images and audio:

```powershell
python extract_oald9.py --media --out-dir extracted_media
```

Extract dictionary records with a limit of X records:

```powershell
python extract_oald9.py --dictionary --limit 10 --out-dir extracted_sample
```

Use a custom data directory:

```powershell
python extract_oald9.py --data-dir D:\path\to\data --dictionary --out-dir extracted_dictionary
```

## Output Layout

Dictionary output:

```text
<out-dir>\dictionary\dic_records_decoded\000000.xml
<out-dir>\dictionary\dic_records_decoded\000001.xml
...
```

Each decoded dictionary XML file contains one or more decoded chunks from the corresponding Lingea record:

```xml
<decoded-record index="0" chunks="34">
  <chunk offset="123" consumed="378"><![CDATA[
<c.entry><c.h-g ...><c.h ...>A <c./h> ...
  ]]></chunk>
</decoded-record>
```

Image output:

```text
<out-dir>\media\images\000001.jpg
<out-dir>\media\images\000002.jpg
...
```

Audio output:

```text
<out-dir>\media\audio\000001.ogg
<out-dir>\media\audio\000002.ogg
...
```

## Diagnostic Output

Use `--diagnostics` when you want extra reverse-engineering views:

```powershell
python extract_oald9.py --dictionary --diagnostics --limit 5 --out-dir extracted_debug
```

This adds:

- `dic_records_readable`: XML wrapping printable text and binary byte runs
- `dic_records_strings`: printable string runs only
- `dic_records_xmlish.txt`: one combined rough printable/XML-like view

These diagnostic files are not needed for normal dictionary extraction.

## Notes

- Full dictionary decoding can take several minutes because `oup_en-dic.lld` contains tens of thousands of records.
- Full audio extraction can produce a very large number of `.ogg` files.
- The extractor intentionally skips unknown media payloads by default and writes only recognized image/audio formats.
- The decoded article files preserve the source's internal tags so later conversion to cleaner XML, HTML, JSON, or a database can be done without losing structure.
