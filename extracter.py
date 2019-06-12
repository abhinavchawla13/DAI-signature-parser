import spacy
import re
import numpy as np
import pprint


def _read_sig(fname):
    return [line.rstrip('\n') for line in open(fname)]

nlp = spacy.load('en_core_web_sm')


# signature = _read_sig('emails/test0_clean.txt')

signature_parsed = {}
signature_unused = []

WEBSITE_PATTERN = "((?:[a-z][\w-]+:(?:\/{1,3}|[a-z0-9%])|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}\/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
EMAIL_PATTERN = "(?:[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*|\"(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21\x23-\x5b\x5d-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])*\")@(?:(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?|\[(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?|[a-z0-9-]*[a-z0-9]:(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21-\x5a\x53-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])+)\])"
PHONE_PATTERN = "(?:(?:\+?([1-9]|[0-9][0-9]|[0-9][0-9][0-9])\s*(?:[.-]\s*)?)?(?:\(\s*([2-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9])\s*\)|([0-9][1-9]|[0-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9]))\s*(?:[.-]\s*)?)?([2-9]1[02-9]|[2-9][02-9]1|[2-9][02-9]{2})\s*(?:[.-]\s*)?([0-9]{4})(?:\s*(?:#|x\.?|ext\.?|extension)\s*(\d+))?"
COMPANY_PATTERN = "(.*(AB|AG|SA|SARL|GMBH|BVBA|Ltée|ltée|LLP|llp|Ltd|ltd|Inc|inc|Corp|corp).*)"
company_found_via_regex = False
parsed_indexes = {}

def _parse_to_signature_dict(type, doc, index, regex_priority=False):
    if (not type in signature_parsed or regex_priority):
        extract_function = "_extract_" + type
        parsed_data = globals()[extract_function](doc, index=index)
        if (parsed_data):
            signature_parsed[type] = parsed_data.rstrip("\n")
            parsed_indexes[type] = index
            return True    


def _extract_name(doc, index):
    # look if the entity is of type PERSON
    split_line = str(doc).split(",", 1)

    # check splitting if name is followed by something in same line
    for entity in doc.ents:
        if (entity.label_ == 'PERSON'):
            if (len(split_line) > 1):
                signature.insert(index + 1, split_line[1])
            return split_line[0]

    # look if the words are proper nouns
    if (len(doc) >= 2):
        if (doc and doc[0] and doc[0].pos_ and doc[0].pos_ == 'PROPN' and
            doc[1] and doc[1].pos_ and doc[1].pos_ == 'PROPN' and 
            not 'name' in signature_parsed):
            return doc

def _extract_company(doc, index):

    global company_found_via_regex
    if (company_found_via_regex): return
    company_from_regex = re.findall(COMPANY_PATTERN, str(doc))
    if (company_from_regex):
        company_found_via_regex = True
        return company_from_regex[0][0]

            
    num_org = np.sum([(entity.label_ == "ORG") for entity in doc.ents])
    if (len(doc) and float(num_org) / len(doc) >= 0.5):
        company_found_via_regex = False
        return doc.text


def _extract_website(doc, index):
    website_match = re.search(WEBSITE_PATTERN, doc)
    if (website_match):
        return doc[website_match.start():website_match.end()]

def _extract_phone(doc, index):
    phone_match = re.search(PHONE_PATTERN, doc)
    if (phone_match):
        return doc[phone_match.start():phone_match.end()]

def _extract_email(doc, index):
    email_match = re.search(EMAIL_PATTERN, doc)
    if (email_match):
        return doc[email_match.start():email_match.end()]

def _extract_title(line):
    doc = nlp(re.sub("[,.-]", "", line.lower()))
    adj_noun_count = np.sum([(token.pos_ == "ADJ" or token.pos_ == "NOUN") for token in doc])
    current_probability = 0
    if (len(doc) > 0):
        current_probability = float(adj_noun_count) / len(doc)
    root = list(filter(lambda token: token.dep_ == 'ROOT', doc))[0]
    # if ROOT is NOUN and prob >= 0.5
    if ((root.pos_ == 'NOUN' or root.pos_ == 'VERB') and current_probability >= 0.5):
        return line.strip(".,")
    # if ROOT is not NOUN and prob > 0.8
    if (current_probability > 0.8):
        return line.strip(".,")

def _extract_address(unparsed):
    UK_POSTAL_CODE = "([Gg][Ii][Rr] 0[Aa]{2})|((([A-Za-z][0-9]{1,2})|(([A-Za-z][A-Ha-hJ-Yj-y][0-9]{1,2})|(([A-Za-z][0-9][A-Za-z])|([A-Za-z][A-Ha-hJ-Yj-y][0-9][A-Za-z]?))))\s?[0-9][A-Za-z]{2})"
    US_POSTAL_CODE = "[0-9]{5}(?:-[0-9]{4})?"
    CA_POSTAL_CODE = "[A-Za-z]\d[A-Za-z][ -]?\d[A-Za-z]\d"
    IN_POSTAL_CODE = "([0-9]{6}|[0-9]{3}\s[0-9]{3})"

    predicted_address = []
    postal_code_found = False
    GP_found = False

    last_index_noted = None
    for i in reversed(range(len(unparsed))):
        if (not postal_code_found or GP_found):
            if (re.search(UK_POSTAL_CODE, unparsed[i]) or
                re.search(US_POSTAL_CODE, unparsed[i]) or
                re.search(CA_POSTAL_CODE, unparsed[i]) or
                re.search(IN_POSTAL_CODE, unparsed[i])):
                predicted_address.append(unparsed[i])
                postal_code_found = True
                last_index_noted = i
            else:
                doc = nlp(unparsed[i])
                if(np.sum([entity.label_ == 'GPE' for entity in doc.ents]) > 0):
                    predicted_address.append(unparsed[i])
                    GP_found = True
                    last_index_noted = i
    
    if (last_index_noted and last_index_noted > 0):
        if (unparsed[last_index_noted - 1] and re.search('\d', unparsed[last_index_noted - 1])
            and not re.search(PHONE_PATTERN, unparsed[last_index_noted - 1])
            and not re.search(EMAIL_PATTERN, unparsed[last_index_noted - 1])):
            predicted_address.insert(0, unparsed[last_index_noted - 1])

    if (len(predicted_address) == 1):
        return predicted_address[0].rstrip("\n")
    elif (len(predicted_address) == 2):
        return predicted_address[0].rstrip("\n") + ", " + predicted_address[1].rstrip("\n")
    
    return

def extract(signature):
    i = 0
    while (i < len(signature)):
        doc = nlp(signature[i])

        _parse_to_signature_dict('name', doc, index=i)
        _parse_to_signature_dict('company', doc, index=i, regex_priority=True)
        _parse_to_signature_dict('website', str(doc).lower(), index=i)
        _parse_to_signature_dict('phone', str(doc).lower(), index=i)
        _parse_to_signature_dict('email', str(doc).lower(), index=i)

        i += 1

    unparsed = _collect_unparsed(signature)
    # _print_unparsed(unparsed)

    further_unparsed = []
    for i in range(len(unparsed)):
        
        title = _extract_title(unparsed[i])
        if (title and not 'title' in signature_parsed):
            signature_parsed['title'] = title.rstrip("\n")
            continue
        
        further_unparsed.append(unparsed[i])

    # _print_unparsed(further_unparsed)

    address = _extract_address(further_unparsed)
    if (address):
        signature_parsed['address'] = address.rstrip("\n")

    return signature_parsed

def _collect_unparsed(signature):
    unparsed = []
    global parsed_indexes
    parsed_indexes_values = parsed_indexes.values()
    # ignore lines seen before
    # ignore lines before name
    for i in range(max(min(parsed_indexes_values), 0), len(signature)):
        if (i not in parsed_indexes_values and signature[i]):
            unparsed.append(signature[i])
    return unparsed

def _print_unparsed(unparsed):
    for line in unparsed:
        print(line + "\n")

def _print_unparsed(unparsed):
    for line in unparsed:
        print(line + "\n")

# main()
# print("---------------------------------------------")
# pprint.pprint(signature_parsed)
# print("---------------------------------------------")



# doc = nlp(u'Montreal')

# for token in doc:

# for q in doc.ents:
