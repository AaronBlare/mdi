def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

def load_subject_info(fn):
    print('\n\n')
    f = open(fn)
    key_line = f.readline()
    keys = key_line.split('\t')
    keys = [x.rstrip() for x in keys]

    subject_info_dict = {}
    for key in keys:
        subject_info_dict[key] = []

    for line in f:
        values = line.split('\t')
        for key_id in range(0, len(keys)):
            key = keys[key_id]
            value = values[key_id].rstrip()
            if key != 'CODE':
                if is_float(value):
                    subject_info_dict[key].append(float(value))
                else:
                    subject_info_dict[key].append(value)
            else:
                subject_info_dict[key].append(value)
    f.close()

    return subject_info_dict


def T0_T1_subject_separation(subject_info_dict):
    T0_dict = {}
    T1_dict = {}

    for key in subject_info_dict:
        T0_dict[key] = []
        T1_dict[key] = []

    time_key = 'time'
    for id, time in enumerate(subject_info_dict[time_key]):
        if time == 'T0':
            for key in subject_info_dict:
                T0_dict[key].append(subject_info_dict[key][id])
        elif time == 'T1':
            for key in subject_info_dict:
                T1_dict[key].append(subject_info_dict[key][id])

    print(f'Number of subjects in T0:{len(T0_dict[time_key])}')
    print(f'Number of subjects in T1:{len(T1_dict[time_key])}')

    return T0_dict, T1_dict
