"""Convert Semeval ABSA gold XML files into TSV training/test inputs."""
import xml.etree.ElementTree as ET

POLARITY_MAP = {'positive': 1, 'neutral': 0, 'negative': -1}
CATEGORY_MAP = {
    'food': 0, 'drinks': 0,
    'restaurant': 1, 'ambience': 1, 'location': 1,
    'service': 2,
}
_CATEGORY_NAMES = {0: 'food', 1: 'place', 2: 'service'}


def semeval_to_csv(f_in: str, f_out: str, multi: bool = True):
    """Parse Semeval gold XML and write rows of ``idx cat polarity separated_sentence``.

    The separated sentence embeds the target span between ``[SEP]`` markers so
    downstream loaders can split (left context, target, right context).
    """
    root = ET.parse(f_in).getroot()
    act_cnt = 0
    cntr = -1
    polarity_cnt = {name: 0 for name in ('positive', 'neutral', 'negative')}
    category_cnt = {name: 0 for name in _CATEGORY_NAMES.values()}

    with open(f_out, 'w', encoding='utf-8') as file:
        for sentence in root.iter('sentence'):
            sent = sentence.find('text').text

            op_cntr = 0
            for opinion in sentence.iter('Opinion'):
                op_cntr += 1

            if op_cntr > 1 and not multi:
                continue

            for opinion in sentence.iter('Opinion'):
                sentiment = opinion.get('polarity')
                polarity = POLARITY_MAP.get(sentiment)

                aspect = opinion.get('category').lower().split('#')[0]
                category = CATEGORY_MAP.get(aspect)

                if category is None or polarity is None:
                    continue

                cntr += 1
                polarity_cnt[sentiment] += 1
                category_cnt[_CATEGORY_NAMES[category]] += 1

                start = int(opinion.get('from'))
                end = int(opinion.get('to'))
                sep_sentence = f'{sent[:start]} [SEP] {sent[start:end]} [SEP] {sent[end:]}'

                act_cnt += 1
                file.write(f"{cntr}\t{category}\t{polarity}\t{sep_sentence}\n")

    print(act_cnt)
    print(polarity_cnt)
    print(category_cnt)


if __name__ == '__main__':
    semeval_to_csv(r'datasets\restaurant\EN_REST_SB1_TEST.xml.gold', 'test_multi16.txt', multi=True)
    semeval_to_csv(r'datasets\restaurant\ABSA15_Restaurants_Test.xml', 'test_multi15.txt', multi=True)
