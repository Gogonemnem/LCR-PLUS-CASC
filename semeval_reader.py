from ssl import OP_NO_COMPRESSION
import xml.etree.ElementTree as ET

def semeval_to_csv(f_in: str, f_out: str, multi=True):
    root = ET.parse(f_in).getroot()
    cntr = -2

    with open(f_out, 'w', encoding='utf-8') as file:

        for sentence in root.iter('sentence'):
            cntr += 1
            sent = sentence.find('text').text
            
            op_cntr = 0
            categories = dict()
            for opinion in sentence.iter('Opinion'):
                op_cntr += 1
                
                target = opinion.get('target')
                if target in categories:
                    categories[target] += 1
                else:
                    categories[target] = 1

            if op_cntr > 1 and not multi:
                continue
                        
            for opinion in sentence.iter('Opinion'):
                cntr += 1
                target = opinion.get('target')
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

                sentence = f'{sent[:start]} [SEP] {sent[start:end]} [SEP] {sent[end:]}'
                line = f"{str(cntr).ljust(4, ' ')}\t{category}\t{polarity}\t{sentence}\n"
                file.write(line)
                # writer.writerow({"context_left": context_left, "target": sent[start:end], "context_right": context_right, "polarity": polarity})
        file.write(str(cntr))

if __name__ ==  "__main__":
    # semeval_to_csv('ABSA16_Restaurants_Train_SB1_v2.xml' , 'val_single.txt', multi=False)
    # semeval_to_csv('ABSA16_Restaurants_Train_SB1_v2.xml' , 'val_multi.txt', multi=True)

    # semeval_to_csv('EN_REST_SB1_TEST.xml.gold' , 'test_single.txt', multi=False)
    # semeval_to_csv('EN_REST_SB1_TEST.xml.gold' , 'test_multi.txt', multi=True)

    semeval_to_csv('ABSA-15_Restaurants_Train_Final.xml' , 'val_single.txt', multi=False)
    semeval_to_csv('ABSA-15_Restaurants_Train_Final.xml' , 'val_multi.txt', multi=True)

    semeval_to_csv('ABSA15_Restaurants_Test.xml' , 'test_single.txt', multi=False)
    semeval_to_csv('ABSA15_Restaurants_Test.xml' , 'test_multi.txt', multi=True)