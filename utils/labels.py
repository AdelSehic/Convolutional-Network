class_to_name = {
    'A1': 'SU-35',
    'A2': 'C-130 Hercules',
    'A3': 'C-17 Globemaster',
    'A4': 'C-5 Galaxy',
    'A5': 'F-4 Phantom',
    'A6': 'TU-160',
    'A7': 'E-3 Sentry',
    'A8': 'KC-135 Stratotanker',
    'A9': 'C-130 Variant(?)',
    'A10': 'F-111',
    'A11': 'E-8C',
    'A12': 'MiG-27',
    'A13': 'F-15E Strike Eagle',
    'A14': 'Airbus',
    'A15': 'F-35',
    'A16': 'F/A-18',
    'A17': 'TU-95',
    'A18': 'Boeing 747',
    'A19': 'SU-27',
    'A20': 'SU-24'
}

def translate_labels(data):
    for i in range(len(data)):
        data[i] = class_to_name[data[i]]
