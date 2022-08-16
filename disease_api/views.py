import numpy as np
import pandas as pd
from .apps import DiseaseApiConfig
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

columns = [
"itching",
"skin_rash",
"nodal_skin_eruptions",
"continuous_sneezing",
"shivering",
"chills",
"joint_pain",
"stomach_pain",
"acidity",
"ulcers_on_tongue",
"muscle_wasting",
"vomiting",
"burning_micturition",
"spotting_ urination",
"fatigue",
"weight_gain",
"anxiety",
"cold_hands_and_feets",
"mood_swings",
"weight_loss",
"restlessness",
"lethargy",
"patches_in_throat",
"irregular_sugar_level",
"cough",
"high_fever",
"sunken_eyes",
"breathlessness",
"sweating",
"dehydration",
"indigestion",
"headache",
"yellowish_skin",
"dark_urine",
"nausea",
"loss_of_appetite",
"pain_behind_the_eyes",
"back_pain",
"constipation",
"abdominal_pain",
"diarrhoea",
"mild_fever",
"yellow_urine",
"yellowing_of_eyes",
"acute_liver_failure",
"fluid_overload",
"swelling_of_stomach",
"swelled_lymph_nodes",
"malaise",
"blurred_and_distorted_vision",
"phlegm",
"throat_irritation",
"redness_of_eyes",
"sinus_pressure",
"runny_nose",
"congestion",
"chest_pain",
"weakness_in_limbs",
"fast_heart_rate",
"pain_during_bowel_movements",
"pain_in_anal_region",
"bloody_stool",
"irritation_in_anus",
"neck_pain",
"dizziness",
"cramps",
"bruising",
"obesity",
"swollen_legs",
"swollen_blood_vessels",
"puffy_face_and_eyes",
"enlarged_thyroid",
"brittle_nails",
"swollen_extremeties",
"excessive_hunger",
"extra_marital_contacts",
"drying_and_tingling_lips",
"slurred_speech",
"knee_pain",
"hip_joint_pain",
"muscle_weakness",
"stiff_neck",
"swelling_joints",
"movement_stiffness",
"spinning_movements",
"loss_of_balance",
"unsteadiness",
"weakness_of_one_body_side",
"loss_of_smell",
"bladder_discomfort",
"foul_smell_of urine",
"continuous_feel_of_urine",
"passage_of_gases",
"internal_itching",
"toxic_look_(typhos)",
"depression",
"irritability",
"muscle_pain",
"altered_sensorium",
"red_spots_over_body",
"belly_pain",
"abnormal_menstruation",
"dischromic _patches",
"watering_from_eyes",
"increased_appetite",
"polyuria",
"family_history",
"mucoid_sputum",
"rusty_sputum",
"lack_of_concentration",
"visual_disturbances",
"receiving_blood_transfusion",
"receiving_unsterile_injections",
"coma",
"stomach_bleeding",
"distention_of_abdomen",
"history_of_alcohol_consumption",
"fluid_overload.1",
"blood_in_sputum",
"prominent_veins_on_calf",
"palpitations",
"painful_walking",
"pus_filled_pimples",
"blackheads",
"scurring",
"skin_peeling",
"silver_like_dusting",
"small_dents_in_nails",
"inflammatory_nails",
"blister",
"red_sore_around_nose",
"yellow_crust_ooze",
]


disease = [
"prognosis_(vertigo) Paroymsal  Positional Vertigo",
"prognosis_AIDS",
"prognosis_Acne",
"prognosis_Alcoholic hepatitis",
"prognosis_Allergy",
"prognosis_Arthritis",
"prognosis_Bronchial Asthma",
"prognosis_Cervical spondylosis",
"prognosis_Chicken pox",
"prognosis_Chronic cholestasis",
"prognosis_Common Cold",
"prognosis_Dengue",
"prognosis_Diabetes ",
"prognosis_Dimorphic hemmorhoids(piles)",
"prognosis_Drug Reaction",
"prognosis_Fungal infection",
"prognosis_GERD",
"prognosis_Gastroenteritis",
"prognosis_Heart attack",
"prognosis_Hepatitis B",
"prognosis_Hepatitis C",
"prognosis_Hepatitis D",
"prognosis_Hepatitis E",
"prognosis_Hypertension ",
"prognosis_Hyperthyroidism",
"prognosis_Hypoglycemia",
"prognosis_Hypothyroidism",
"prognosis_Impetigo",
"prognosis_Jaundice",
"prognosis_Malaria",
"prognosis_Migraine",
"prognosis_Osteoarthristis",
"prognosis_Paralysis (brain hemorrhage)",
"prognosis_Peptic ulcer diseae",
"prognosis_Pneumonia",
"prognosis_Psoriasis",
"prognosis_Tuberculosis",
"prognosis_Typhoid",
"prognosis_Urinary tract infection",
"prognosis_Varicose veins",
"prognosis_hepatitis A",
]

disease = pd.Index(disease)
# Create your views here.

def prediction(symptoms, classifier, columns, disease) :
  tests = []
  for i in columns :
    if i in symptoms :
        tests.append(1)
    else :
      tests.append(0)
  
  tests = np.array(tests)
  symptoms_test = pd.DataFrame(data = [tests],
                             columns = columns )
  predict = np.argmax(classifier.predict(symptoms_test), axis=1)
  return disease[predict][0]

class DiseaseApiView(APIView):

    # 2. Create
    def post(self, request):
        predictions=""
        data = request.data
        symptoms = data['symptoms']

        model = DiseaseApiConfig.model
        predictions = prediction(symptoms, model, columns, disease)  
        response_dict = {
          "symptoms" : symptoms ,
          "Predicted disease":predictions
          }
        return Response(response_dict, status=200)


