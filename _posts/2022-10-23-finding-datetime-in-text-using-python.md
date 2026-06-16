---
title:  Finding DateTime in Text Using Python
date:   2022-10-23 01:29:17 +0545
categories:
    - Python
    - DateTime
    - Text Processing
tags:
    - python
    - Text Processing
    - DateTime
header:
  teaser: assets/python/date_parser.png
description: "Techniques and Python libraries to extract date and time expressions from free-form text, with practical examples and recommendations."
---

Why extract datetime values from text? In data processing and analysis you often need to identify timestamps embedded in emails, logs, or user messages. Textual date/time formats vary widely, so relying on simple string matching is brittle. This post shows practical approaches with regular expressions and dedicated Python libraries to robustly find datetimes in free-form text.

Below is a sample (simplified) email thread I used for testing:

```python
eml = """Re: Documents Received

John Doe <john@doe.org>
Wed, Jun 1, 2011, 9:39 PM
to Emma, Don, Bucky

Lorem
Ipsum
Dorem

On 01/06/2011, at 7:57 PM, "Emma" <emma@thompson.com> wrote:

Lorem Ipsum?

Thanks John

On 1 June 2011 13:43, Bucky Hallam <bucky@barnes.com> wrote:

Lorem Ipsum is Dorem.

Thanks Emma"""
```

The thread contains dates in different formats:
* Wed, Jun 1, 2011, 9:39 PM
* 01/06/2011, at 7:57 PM (ambiguous: mm/dd vs dd/mm)
* 1 June 2011 13:43

First, a quick demonstration of why naive regexes become tedious.

## Using regular expressions

You can craft regex patterns for specific formats. For example, YYYY/MM/DD:

```python
import re
pattern = r"\d{4}/\d{2}/\d{2}"
txt = "This is 2022/11/11 and we are waiting for 2022/11/12."
print(re.findall(pattern, txt))
# ['2022/11/11', '2022/11/12']
```

To accept both `-` and `/` separators use alternation:

```python
pattern = r"(\d{4}-\d{2}-\d{2}|\d{4}/\d{2}/\d{2})"
```

Including time parts increases complexity. You can keep adding patterns, but maintaining many variants quickly becomes hard. Also, locale-specific formats (day-first vs month-first) and verbose formats like `Wed, Jun 1, 2011, 9:39 PM` are painful to cover exhaustively with regex alone.

## Use a library: python-dateutil

python-dateutil provides a flexible parser. Install with:

    pip install python-dateutil

Example:

```python
from dateutil.parser import parse
parse('1 June 2011 13:43', fuzzy_with_tokens=True)
# (datetime.datetime(2011, 6, 1, 13, 43), ('', ''))
```

dateutil is powerful, but it may fail on very noisy strings or on multiple dates in the same input. It is best when you pass a single candidate substring rather than a whole document containing many dates.

## Use a library: dateparser

dateparser is excellent at handling noisy, human-written date/time expressions and supports settings for languages and day-first vs month-first interpretation. Install with:

    pip install dateparser

Example (searching for dates inside text):

```python
from dateparser.search import search_dates
search_dates(eml)
# [('Wed, Jun 1, 2011, 9:39 PM', datetime.datetime(2011, 6, 1, 21, 39)),
#  ('On 01/06/2011, at 7:57 PM', datetime.datetime(2011, 1, 6, 19, 57)),
#  ('On 1 June 2011 13:43', datetime.datetime(2011, 6, 1, 13, 43))]
```

Note: `dateparser` interpreted `01/06/2011` as month/day by default in this example. Use settings to disambiguate:

```python
from dateparser.search import search_dates
from dateparser import parse
# Force day-first
search_dates(eml, settings={'DATE_ORDER': 'DMY'})
```

dateparser returns both the matched substring and a Python datetime object, which makes it practical for splitting text into segments based on the original text.

## Use a library: datefinder

datefinder is another option that yields datetime objects for many common patterns.

    pip install datefinder

Example:

```python
from datefinder import find_dates
list(find_dates(eml))
# [datetime.datetime(2011, 6, 1, 21, 39),
#  datetime.datetime(2011, 1, 6, 19, 57),
#  datetime.datetime(2011, 6, 1, 13, 43)]
```

datefinder is handy when you only need datetime objects and are less concerned about preserving the exact matched text format.

## Which tool to choose?

* If you need a robust search over noisy text and want the original matched substring + datetime object: use dateparser.search.search_dates and configure settings (language, DATE_ORDER).
* If you already have a candidate substring or a consistent format: python-dateutil.parse is reliable and fast.
* If you only need datetime objects and accept some ambiguity: datefinder can be convenient.

## Practical tips
* Be aware of ambiguous numeric formats (01/06/2011). Explicitly set DATE_ORDER or try to detect locale first.
* Use the library's settings to control time zones and languages when applicable.
* When processing large documents, first narrow down candidate regions with lightweight regexes (e.g., lines containing month names or numbers and AM/PM) and then pass candidates to a parser.
* Persist both the parsed datetime and the original matched substring if you need to preserve the original text for display or auditing.

## Conclusion

For my use case (parsing email threads) `dateparser` performed best because it finds multiple date expressions in noisy, conversational text and returns Python datetime objects along with the original substrings. Try small samples from your own data and compare results from multiple libraries before committing to one.

For more posts like this, explore the site or subscribe to the [newsletter](https://dataqoil.com/newsletter/).



