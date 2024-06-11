import sys, time
sys.path.append("NL-Augmenter")

# pip install spacy torchtext cucco fastpunct sacremoses
# python -m spacy download en_core_web_sm


from nlaugmenter.transformations.butter_fingers_perturbation.transformation import ButterFingersPerturbation
from nlaugmenter.transformations.random_deletion.transformation import RandomDeletion
from nlaugmenter.transformations.synonym_substitution.transformation import SynonymSubstitution
from nlaugmenter.transformations.back_translation.transformation import BackTranslation
from nlaugmenter.transformations.change_char_case.transformation import ChangeCharCase
from nlaugmenter.transformations.whitespace_perturbation.transformation import WhitespacePerturbation
from nlaugmenter.transformations.underscore_trick.transformation import UnderscoreTrick
from nlaugmenter.transformations.style_paraphraser.transformation import StyleTransferParaphraser
from nlaugmenter.transformations.punctuation.transformation import PunctuationWithRules




def aug_generator(text_list, aug_style):

    if aug_style == "butter_fingers":
        t1 = ButterFingersPerturbation(max_outputs=1)
        return [t1.generate(text_list[i], prob = 0.1)[0] for i in range(len(text_list))]
    elif aug_style == "random_deletion":
        t1 = RandomDeletion(prob=0.25)
        return [t1.generate(text_list[i])[0] for i in range(len(text_list))]
    elif aug_style == "synonym_substitution":
        syn = SynonymSubstitution(max_outputs=1, prob = 0.2)
        return [syn.generate(text_list[i])[0] for i in range(len(text_list))]
    elif aug_style == "back_translation":
        trans = BackTranslation()
        return [trans.generate(text_list[i])[0] for i in range(len(text_list))]
    elif aug_style == "change_char_case":
        t1 = ChangeCharCase()
        return [t1.generate(text_list[i], prob = 0.25)[0] for i in range(len(text_list))]
    elif aug_style == "whitespace_perturbation":
        t1 = WhitespacePerturbation()
        return [t1.generate(text_list[i], prob = 0.25)[0] for i in range(len(text_list))]
    elif aug_style == "underscore_trick":
        t1 = UnderscoreTrick(prob = 0.25)
        return [t1.generate(text_list[i])[0] for i in range(len(text_list))]
    elif aug_style == "style_paraphraser":
        t1 = StyleTransferParaphraser(style = "Basic", upper_length="same_5")
        return [t1.generate(text_list[i])[0] for i in range(len(text_list))]
    elif aug_style == "punctuation_perturbation":
        normalizations = ['remove_extra_white_spaces', ('replace_characters', {'characters': 'was', 'replacement': 'TZ'}),
                      ('replace_emojis', {'replacement': 'TESTO'})]
        punc = PunctuationWithRules(rules=normalizations)
        return [punc.generate(text_list[i])[0] for i in range(len(text_list))]
    else:
        raise ValueError("Augmentation style not found. Please check the available styles.")

def generate_perturbations(text_list):
    augmentation_styles = ["synonym_substitution", "butter_fingers", "random_deletion", "change_char_case", "whitespace_perturbation", "underscore_trick"]
    all_augmented = {}
    for style in augmentation_styles:
        start = time.time()
        aug_list = aug_generator(text_list, style)
        all_augmented[style] = aug_list
        print(f"Perturbing with {style} took {time.time() - start} seconds")
    return all_augmented
        
if __name__ == "__main__":
    text_list = ["This is a test sentence. It is a good sentence.", "This is another test sentence. It is a bad sentence."]
    print(generate_perturbations(text_list))