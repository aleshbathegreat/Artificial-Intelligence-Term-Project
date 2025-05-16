#assuming valence and arousal values
#remember to cite simpful
from simpful import *

class EmotionLogic:

    def fuzzy(self, ars, val):
        fuzz = FuzzySystem()
        trianglemembership1 = AutoTriangle(3, terms = ['negative', 'neutral', 'positive'], universe_of_discourse = [1,9])
        trianglemembership2 = AutoTriangle(3, terms = ['calm', 'neutral', 'excited'], universe_of_discourse = [1,9])
        fuzz.add_linguistic_variable("valence", trianglemembership1)
        fuzz.add_linguistic_variable("arousal", trianglemembership2)

        #numerical emotion values, terms
        #since suegmo gives only crisp float vals, we need to map functions to numbers
        fuzz.set_crisp_output_value("sad", 1)
        fuzz.set_crisp_output_value("bored", 2)
        fuzz.set_crisp_output_value("angry", 3)
        fuzz.set_crisp_output_value("drowsy", 4)
        fuzz.set_crisp_output_value("calm", 5)
        fuzz.set_crisp_output_value("excited", 6)
        fuzz.set_crisp_output_value("relaxed", 7)
        fuzz.set_crisp_output_value("pleasant", 8)
        fuzz.set_crisp_output_value("happy", 9)

        rules_set = [
            "IF (valence IS negative) AND (arousal IS calm) THEN (emotion IS sad)",
            "IF (valence IS negative) AND (arousal IS neutral) THEN (emotion IS bored)",
            "IF (valence IS negative) AND (arousal IS excited) THEN (emotion IS angry)",
            "IF (valence IS neutral) AND (arousal IS calm) THEN (emotion IS drowsy)",
            "IF (valence IS neutral) AND (arousal IS neutral) THEN (emotion IS calm)",
            "IF (valence IS neutral) AND (arousal IS excited) THEN (emotion IS excited)",
            "IF (valence IS positive) AND (arousal IS calm) THEN (emotion IS relaxed)",
            "IF (valence IS positive) AND (arousal IS neutral) THEN (emotion IS pleasant)",
            "IF (valence IS positive) AND (arousal IS excited) THEN (emotion IS happy)"
        ]
        fuzz.add_rules(rules_set)
        
        fuzz.set_variable("valence", val)
        fuzz.set_variable("arousal", ars)

        emotion = fuzz.Sugeno_inference()
        emo_val = emotion["emotion"]
        

        emotion_map = {
                    1 : 'SAD',
                    2 : 'BORED',
                    3 :  'ANGRY',
                    4 :  'DROWSY',
                    5 :  'CALM',
                    6 :  'EXCITED',
                    7 : 'RELAXED',
                    8 : 'PLEASANT',
                    9 : 'HAPPY'
        }
        
        return emotion_map[int(emo_val)]

# if __name__ == "__main__":
#     Logic = EmotionLogic()
#     emotion = Logic.fuzzy(1.2, 1.3)
#     print(emotion)
