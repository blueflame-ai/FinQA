# Reviewed DocFinQA corrections, version 1

This is an **opt-in corrected DocFinQA export**, hosted in the BlueFlame fork of
[FinQA](https://github.com/czyssrs/FinQA). It corrects 109 records reviewed in a
200-question sample of the DocFinQA training split. It is not a repair of the
entire FinQA or DocFinQA corpus, and its scores must not be reported as scores on
the unchanged official benchmark.

The historical `dataset/{train,dev,test,private_test}.json` files remain unchanged.
The corrected export uses DocFinQA's four fields (`Context`, `Question`, `Program`,
`Answer`); it is not input for the original FinQANet DSL evaluator.

## What changed

- **38 source contexts:** 11 records contained the wrong reporting year or stock
  graph base period. These now use the complete correct-year report. Two other
  omitted-table records use complete replacement reports. Twenty-five records
  restore missing source tables or disclosures at exact source positions.
- **References:** fix arithmetic, base years, denominators, signs, share and money
  scales, percent versus percentage points, and mistaken financial quantities.
  For example, the mean of 162, 149 and 145 is 152, not 151; 855/775.2 is 1.103
  to three decimals, not 1.14; and aggregate redeemed par value of 2.2 million
  must not be multiplied by the number of units a second time.
- **Questions:** identify the intended year, entity, segment, benefit plan,
  acquisition, or asset class when expanding a page-level FinQA question to a
  full report made it ambiguous. Restore the missing category in the Magma
  purchase-allocation question. State units, direction and rounding explicitly.
- **Programs:** supply a matching Python arithmetic reference for every corrected
  answer. These are reference calculations, not a required method for answering.
  The application tool never executes them.

The two Motorola references mix Home and Networks Mobility segment sales with
Mobile Devices customer concentration. The aggregate-sales question now explicitly
asks for consolidated 2008 net sales (30,146 million dollars). The separate
customer-sales question names Home and Networks Mobility and uses that segment's
own 2008 concentration of 45 percent: 10,086 × 45% = 4,538.7 million dollars.
The old 41% and 42% bases belong to a separate segment and must not be carried
across the adjacent heading.

Questions with unsupported premises are made answerable from the report: for
example, Citi's goodwill is not allocated book value, a bank's own-account share
sale is not issuer financing cash flow, and an unspecified number of conduits
cannot support a per-conduit average. These are explicitly new question contracts;
they do not retroactively determine whether a previous model response passed.

## Which upstream layer was wrong?

DocFinQA expands FinQA's page excerpts into full documents. The eleven wrong-year
contexts and the omitted tables/disclosures are present in the public DocFinQA
records. The corresponding original FinQA IDs identify the intended report year
and preserve the needed page evidence. Replacing these contexts fixes the public
DocFinQA conversion; it does not imply that original FinQA supplied the wrong PDF.

Several reference defects originate in FinQA itself: its human `qa.answer`,
`qa.exe_ans`, and `qa.program` can disagree. The correction manifest preserves the
original human answer, execution answer, table and gold evidence alongside each
new reference, so reviewers can distinguish inherited errors from conversion
errors. Full-document ambiguity is a third issue: a question that was scoped by
its original page may need an explicit segment or plan after expansion.

The sample's remaining 91 records are unchanged. Genuine wrong-table or wrong-
entity model answers are not repaired by changing their targets. No models were
rerun, no previous scores were rewritten, and no application-specific judge was
modified. A grader still needs to normalize units and respect the question's
rounding contract; these data changes do not fix a grader that contradicts an
otherwise correct answer.

## Provenance and reproducibility

The manifest is
[`dataset/corrections/docfinqa-v1/corrections.json`](../dataset/corrections/docfinqa-v1/corrections.json).
It pins:

- FinQA revision `0f16e2867befa6840783e58be38c9efb9229d742`.
- [Public DocFinQA](https://huggingface.co/datasets/kensho/DocFinQA) revision
  `64ebaff62f692495bcc182f45cf9a9606251b19b`.
- Each exact original question, answer, program, and SHA-256 of UTF-8 context.
- Original FinQA report/page IDs and the unchanged original evidence.
- Complete replacement report text, source PDF URL, PDF SHA-256, extracted-text
  SHA-256, and extraction command. The PDF hashes identify the source downloads;
  the application verifies the bundled extracted text.

Bundled report texts were extracted with Poppler `pdftotext` in default reading order. Form-feed
page boundaries and complete reports are retained. They are public source
financial reports, not model outputs. Restored tables come from the original
FinQA `table_ori`; restored narrative disclosures come from its source-page text.
Restored narrative punctuation is normalized for readability. No question, gold label, or correction rationale is appended to a model context.

Export the pinned public training split to JSONL using Hugging Face Datasets in
a separate environment that has `datasets` installed:

```python
from datasets import load_dataset

train = load_dataset(
    "kensho/DocFinQA",
    revision="64ebaff62f692495bcc182f45cf9a9606251b19b",
    split="train",
)
train.select_columns(["Context", "Question", "Program", "Answer"]).to_json("docfinqa-train.jsonl")
```

Apply the overlay with Python 3.10 or later; the tool itself has no third-party
dependencies or network calls:

```sh
python code/apply_docfinqa_corrections.py \
  --input docfinqa-train.jsonl \
  --output docfinqa-train-corrected-v1.jsonl
python -m unittest discover -s tests -v
```

Input can be the full training split or a subset containing all 109 patched
records. All other records pass through unchanged. Every patched record must
match exactly once. A changed context, reference, duplicate, missing record, or
nonunique insertion anchor aborts the operation. New facts outside the original
FinQA page evidence also have required context excerpts, checked after repair. Output is replaced atomically
only after validation succeeds. Reviewers should keep the manifest and its
reference/evidence fields outside the model-visible input.

Local verification applied the overlay to all 200 original public sample
records, checked all 109 exact reference/context contracts and 38 context repairs,
and confirmed the other 91 records were unchanged. CI validates the manifest
against the original FinQA evidence, computes all reference programs, checks all
bundled source hashes, and exercises drift, duplicate, partial-input, source
integrity and atomic-output guards without downloading data or calling a model.
The full public JSONL is intentionally not bundled; reproducing that integration
check requires the pinned upstream export.
