"""Convert Semeval ABSA gold XML files into TSV training/test inputs."""
import os
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

    os.makedirs(os.path.dirname(os.path.abspath(f_out)), exist_ok=True)
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
    # Resolve the active domain's data root (falls back to restaurant), then
    # emit both label_type splits for each SemEval test gold into the
    # {root}/{year}/test_{single,multiple}.txt layout that data.load_semeval reads.
    try:
        from ..config import domain
        root = domain().root_path
    except Exception:
        root = 'datasets/restaurant'

    golds = {
        2016: f'{root}/raw/EN_REST_SB1_TEST.xml.gold',
        2015: f'{root}/raw/ABSA15_Restaurants_Test.xml',
    }
    for year in sorted(golds):
        f_in = golds[year]
        if not os.path.exists(f_in):
            print(f'skip {year}: no gold {f_in}')
            continue
        os.makedirs(f'{root}/{year}', exist_ok=True)
        semeval_to_csv(f_in, f'{root}/{year}/test_single.txt', multi=False)
        semeval_to_csv(f_in, f'{root}/{year}/test_multiple.txt', multi=True)
