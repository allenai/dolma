import unittest

from dolma.core.data_types import InputSpecWithMetadata
from dolma.taggers.licenses import CreativeCommonsRegexLicenseExtractor

LICENSES = [
    ('<a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>', "by", "4.0", "null"),
    (
        '<p xmlns:cc="http://creativecommons.org/ns#" xmlns:dct="http://purl.org/dc/terms/"><a property="dct:title" rel="cc:attributionURL" href="http://test.com/pic.jpg">Test</a> by <a rel="cc:attributionURL dct:creator" property="cc:attributionName" href="http://test.com">Dude</a> is marked with <a href="https://creativecommons.org/publicdomain/zero/1.0/?ref=chooser-v1" target="_blank" rel="license noopener noreferrer" style="display:inline-block;">CC0 1.0<img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/cc.svg?ref=chooser-v1" alt=""><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/zero.svg?ref=chooser-v1" alt=""></a></p>',  # noqa
        "publicdomain/zero",
        "1.0",
        "null",
    ),
    (
        '<a href="http://creativecommons.org/licenses/by/4.0/?ref=chooser-v1" target="_blank" rel="license noopener noreferrer" style="display:inline-block">CC BY 4.0</a>',  # noqa
        "by",
        "4.0",
        "null",
    ),
    (
        '<p xmlns:cc="http://creativecommons.org/ns#" xmlns:dct="http://purl.org/dc/terms/"><a property="dct:title" rel="cc:attributionURL" href="http://test.com/pic.jpg">Test</a> by <a rel="cc:attributionURL dct:creator" property="cc:attributionName" href="http://test.com">Dude</a> is licensed under <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/?ref=chooser-v1" target="_blank" rel="license noopener noreferrer" style="display:inline-block;">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International<img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/cc.svg?ref=chooser-v1" alt=""><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/by.svg?ref=chooser-v1" alt=""><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/nc.svg?ref=chooser-v1" alt=""><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/sa.svg?ref=chooser-v1" alt=""></a></p>',  # noqa
        "by-nc-sa",
        "4.0",
        "null",
    ),
    (
        '<a href="https://creativecommons.org/licenses/by/3.0/">CC BY 3.0</a>',
        "by",
        "3.0",
        "null",
    ),
    (
        'This page, by <a href="http://lessig.org/">Lawrence Lessig</a>, is licensed under a <a rel="license" href="https://creativecommons.org/licenses/by/3.0/"> Creative Commons Attribution License</a>.',  # noqa
        "by",
        "3.0",
        "null",
    ),
    (
        '<div>License: <a href="https://creativecommons.org/licenses/by-nc/4.0/">Creative Commons Attribution-NonCommercial 4.0 International</a></div>',
        "by-nc",
        "4.0",
        "null",
    ),
    (
        '<a href="https://creativecommons.org/licenses/by/4.0/deed.es" hreflang="es">Licencia CC BY 4.0</a>',
        "by",
        "4.0",
        "es",
    ),
    (
        '<a href="https://creativecommons.org/licenses/by-nd/3.0/hr/legalcode.hr" hreflang="hr">Licencija CC BY-ND 3.0 HR</a>',
        "by-nd",
        "3.0",
        "hr",
    ),
    (
        '<a href="https://creativecommons.org/licenses/by-nc-sa/2.5/ca/legalcode.en" hreflang="en">CC BY-NC-SA 2.5 Canada License</a>',
        "by-nc-sa",
        "2.5",
        "en",
    ),
]


class CcLicenseTagger(unittest.TestCase):
    def test_license_extraction(self):
        tagger = CreativeCommonsRegexLicenseExtractor()

        for html, gold_type, gold_version, gold_language in LICENSES:
            doc = InputSpecWithMetadata(metadata={"html": html}, id="test", text="")
            out = tagger.tag(doc)

            self.assertEqual(len(out), 1)

            _, pred_type, *rest = list(out)[0].split("_")
            self.assertEqual(pred_type, gold_type)

            if gold_version != "null":
                self.assertGreater(len(rest), 0)
                pred_version, *rest = rest
                self.assertEqual(pred_version, gold_version)

            if gold_language != "null":
                self.assertGreater(len(rest), 0)
                pred_language = rest[0]
                self.assertEqual(pred_language, gold_language)

            # self.assertEqual(len(doc.spans), 1)
            # span = doc.spans[0]
            # self.assertEqual(span.metadata["type"], license_type)
            # self.assertEqual(span.metadata["version"], version)
            # self.assertEqual(span.metadata["lang"], lang)
