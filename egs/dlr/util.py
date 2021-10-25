import os


def get_heirarchy_of_categories():
    return {'Probandengeraeusche': ['Nebengeraeusche'], 'Nachbarschaftslaerm': ['Nebengeraeusche'],
            'Raumknacken': ['Nebengeraeusche'], 'Auto': ['Autos', 'Strassenverkehr'], 'Autos': ['Autos', 'Strassenverkehr'],
            'LKW': ['Grosse Fahrzeuge', 'Strassenverkehr'], 'Motorrad': ['Strassenverkehr'],
            'Transporter': ['Strassenverkehr'], 'Flugzeug_landend': ['Flugzeug'], 'Flugzeug_startend': ['Flugzeug'],
            'Flugzeug': ['Flugzeug'], 'Gueterzug': ['Zuege-Bahnen'], 'Personenzug': ['Zuege-Bahnen'],
            'Straßenbahn': ['Zuege-Bahnen'], 'entgegenkommende_Gueterzug': ['Zuege-Bahnen'],
            'Gueterzug_langsam_fahrend': ['Zuege-Bahnen'], 'entgegenkommende_Personenzug': ['Zuege-Bahnen'],
            'Personenzug_bremsend': ['Zuege-Bahnen'], 'Bahn_Rangierfahrzeug_etc': ['Zuege-Bahnen'],
            'Personenzug_langsam_fahrend': ['Zuege-Bahnen'], 'Güterzug_langsam_fahrend': ['Zuege-Bahnen'],
            'Gueterzug_bremsend': ['Zuege-Bahnen'], 'Messung_Start': [], 'Messung_Ende': [], 'Umdrehen_im_Bett': [],
            'Vogelgezwitscher': [], 'Autobahn': [], 'Fahrzeugkolonne': [], 'lauter_Regen': [], 'Flughafenbodenlärm': [],
            'Aufstehen_Toilettengang_etc': [], 'Husten_Raeuspern': [], 'Sirene_Polizei_Notarzt_Feuerwehr': [],
            'entgegenkommende_Auto': [], 'Wind': [], 'Schnarchen': [], 'Gewitter': [],
            'Grosse Fahrzeuge': ['Grosse Fahrzeuge', 'Strassenverkehr'], 'Strassenverkehr': ['Strassenverkehr'],
            'Zuege-Bahnen': ['Zuege-Bahnen'], 'Silence':['Silence'], 'Nebengeraeusche': ['Nebengeraeusche']}
if __name__ =="__main__":
        count=0
        for key, value in get_heirarchy_of_categories().items():
                folder = '/home/khan_mo/data/final_data'
                dirs = set(os.listdir(folder))
                if key in dirs:
                        #print(key)
                        count+=1
        print(count)
        #print(os.listdir(folder))