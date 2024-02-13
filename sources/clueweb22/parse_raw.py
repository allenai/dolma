import datetime
from typing import Any, Dict, List
from dolma.core.parallel import BaseParallelProcessor, QueueType
import smart_open
from msgspec import field, defstruct
from msgspec.json import Decoder, Encoder


def convert_timestamp(d: datetime.datetime) -> str:
    return d.strftime("%Y-%m-%dT%H:%M:%S.%f")[:23] + "Z"


class Clueweb22RawProcessor(BaseParallelProcessor):
    @classmethod
    def increment_progressbar(cls, queue: "QueueType", /, files: int = 0, documents: int = 0) -> Dict[str, int]:
        return super().increment_progressbar(queue, files=files, documents=documents)

    @classmethod
    def process_single(
        cls,
        source_path: List[str],
        destination_path: str,
        queue: QueueType,
        **kwargs: Any
    ):

        ClueWeb22Spec = defstruct(
            'ClueWeb22Spec',
            (
                ('URL', str, field(name="url")),
                ('URL-hash', str, field(name="url_hash")),
                ('Language', str, field(name="language")),
                ('ClueWeb22-ID', str, field(name="clueweb22_id")),
                ('Clean-Text', str, field(name="clean_text")
            )
        )
        parser = Decoder(ClueWeb22Spec)
        writer = Encoder()

        with smart_open.open(destination_path, "wt") as wf:
            for path in source_path:
                with smart_open.open(path, "rt") as rf:
                    for raw in rf:
                        doc = parser.decode(raw)

                        if re.match(r"clueweb22\-en00*", doc['clueweb22_id']):
                            category = "ClueWeb22-B"
                        elif re.match(r"clueweb22\-en00[1-9]*", doc['clueweb22_id']):
                            category = "ClueWeb22-A"
                        else:
                            category = "ClueWeb22-L"

                        output = {
                            "id": doc['clueweb22_id'],
                            "created":
                            "added": convert_timestamp(datetime.datetime.now()),
                            "source": category,
                            "text": doc['clean_text'],
                            "metadata": {
                                "url": doc['url'],
                                "url_hash": doc['url_hash'],
                                "language": doc['language']
                            }
                        }
                        wf.write(writer(output) + "\n")

        # {
        #     "URL": "https://www.fatsecret.com/calories-nutrition/usda/bananas?portionid=32979&portionamount=1.000\n",
        #     "URL-hash": "5A0F5973A6D7B04E9D59FBCEFB1D8E13",
        #     "Language": "en",
        #     "ClueWeb22-ID": "clueweb22-en0001-00-00000",
        #     "Clean-Text": 'Calories in 1 large Banana and Nutrition Facts\nFoods\nFood List\nBananas\nFood database and calorie counter\n1 large (8" to 8-7/8" long)\nBananas\nNutrition Facts\nServing Size\n1 large (8" to 8-7/8" long)\nAmount Per Serving\nCalories\n121\n% Daily Values*\nTotal Fat\n0.45g\n1%\nSaturated Fat\n0.152g\n1%\nTrans Fat\n-\nPolyunsaturated Fat\n0.099g\nMonounsaturated Fat\n0.044g\nCholesterol\n0mg\n0%\nSodium\n1mg\n0%\nTotal Carbohydrate\n31.06g\n11%\nDietary Fiber\n3.5g\n13%\nSugars\n16.63g\nProtein\n1.48g\nVitamin D\n-\nCalcium\n7mg\n1%\nIron\n0.35mg\n2%\nPotassium\n487mg\n10%\nVitamin A\n4mcg\n0%\nVitamin C\n11.8mg\n13%\n* The % Daily Value (DV) tells you how much a nutrient in a serving of food contributes to a daily diet. 2,000 calories a day is used for general nutrition advice.\nLast updated: 04 Feb 08 05:07 AM\nSource: FatSecret Platform API\n6%\nof RDI*\n(121 calories)\nCalorie Breakdown:\nCarbohydrate (93%)\nFat (3%)\nProtein (4%)\n* Based on a RDI of 2000 calories\nWhat is my Recommended Daily Intake (RDI)?\nPhotos\nview more photos\nNutrition summary:\nCalories\n121\nFat\n0.45g\nCarbs\n31.06g\nProtein\n1.48g\nThere are 121 calories in 1 large Banana.\nCalorie breakdown: 3% fat, 93% carbs, 4% protein.\nOther Common Serving Sizes:\nServing Size\nCalories\n1 oz\n25\n1 extra small (less than 6" long)\n72\n100 g\n89\n1 small (6" to 6-7/8" long)\n90\n1 medium (7" to 7-7/8" long)\n105\n1 NLEA serving\n112\n1 large (8" to 8-7/8" long)\n121\n1 cup sliced\n134\n1 extra large (9" or longer)\n135\n1 cup mashed\n200\nTrader Joe\'s Nothing But Banana Flattened\nGiant Food Bananas (Large)\nBaked Banana\nFried Red Banana\nApple Banana\nview more bananas nutritional info\nRelated Types of Fruit:\nOranges\nRaspberries\nStrawberries\nPeaches\nApples\nview more fruit nutritional info\nSee Also:\nBanana\nDole Bananas\nChiquita  Banana\nDole Baby Banana\nChiquita Mini Banana\nview more results\nUsed in these Member Recipes:\nBanana Oatmeal Muffins\nPeanut Butter Cookies\nBrown Sugar Banana Muffins\nBanana Cottage Cheese Bread with Chocolate Chips\nBanana Muffins with Chocolate and Peanut Butter Chips\nChocolate Banana Protein Muffins\nBanana Muffins\nWaffle\nBanana Oat Cookies\nBanana Oat Protein Cookies',
        # }
