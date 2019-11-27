from config.routines import is_float


def load_cytokines(fn):
    f = open(fn)
    key_line = f.readline()
    keys = key_line.split('\t')
    keys[-1] = keys[-1].rstrip()

    data_dict = {}
    for key in keys:
        data_dict[key] = []

    for line in f:
        if line != '\n':
            values = line.split('\t')
            for key_id in range(0, len(keys)):
                key = keys[key_id]
                value = values[key_id].rstrip()
                if key != 'CODE':
                    if is_float(value):
                        data_dict[key].append(float(value))
                    else:
                        data_dict[key].append(value)
                else:
                    data_dict[key].append(value)
    f.close()

    return data_dict


def T0_T1_cytokines_separation(data_dict):
    T0_dict = {}
    T1_dict = {}

    for key in data_dict:
        T0_dict[key] = []
        T1_dict[key] = []

    time_key = 'time'
    for id, time in enumerate(data_dict[time_key]):
        if time == 'T0':
            for key in data_dict:
                T0_dict[key].append(data_dict[key][id])
        elif time == 'T1':
            for key in data_dict:
                T1_dict[key].append(data_dict[key][id])

    print(f'Number of cytokines in T0:{len(T0_dict[time_key])}')
    print(f'Number of cytokines in T1:{len(T1_dict[time_key])}')

    return T0_dict, T1_dict
