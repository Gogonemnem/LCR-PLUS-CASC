import xml.etree.ElementTree as ET

def semeval_to_csv(f_in: str, f_out: str, multi=True):
    root = ET.parse(f_in).getroot()

    with open(f_out, 'w', encoding='utf-8') as file:

        for sentence in root.iter('sentence'):
            sent = sentence.find('text').text
            
            single = True
            categories = dict()
            for opinion in sentence.iter('Opinion'):
                if single:
                    single = False
                elif not multi:
                    continue
                
                target = sentiment = opinion.get('target')
                if target in categories:
                    categories[target] += 1
                else:
                    categories[target] = 0
                        
            for opinion in sentence.iter('Opinion'):
                target = sentiment = opinion.get('target')
                if categories[target] > 1:
                    continue
                
                sentiment = opinion.get('polarity')
                if sentiment == "positive":
                    polarity = 1
                elif sentiment == "neutral":
                    polarity = 0
                    continue
                elif sentiment == "negative":
                    polarity = -1
                    polarity = 0
                else:
                    polarity = None

                aspect = opinion.get('category').lower().split('#')[0]
                if aspect in ('food', 'drinks'):
                    category = 0
                elif aspect in ('restaurant', 'ambience', 'location'):
                    category = 1
                elif aspect in ('service'):
                    category = 2
                else:
                    category = None

                start = int(opinion.get('from'))
                end = int(opinion.get('to'))

                # skip implicit targets
                if start == end == 0: 
                    continue

                sentence = f'{sent[:start]} [PAD] {sent[start:end]} [PAD] {sent[end:]}'
                line = f'{category} {polarity} {sentence} \n'
                file.write(line)
                # writer.writerow({"context_left": context_left, "target": sent[start:end], "context_right": context_right, "polarity": polarity})

if __name__ ==  "__main__":
    semeval_to_csv('ABSA15_Restaurants_Test.xml' , 'test.txt')