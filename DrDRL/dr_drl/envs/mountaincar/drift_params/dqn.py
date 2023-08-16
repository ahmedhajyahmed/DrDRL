"""
Taken from
https://github.com/testingautomated-usi/rl-plasticity-experiments
"""
adaptable_env = [
    {'force': 0.001, 'goal_velocity': 0.0525, 'gravity': 0.0025},
    {'force': 0.0009262000000000001, 'goal_velocity': 0.0, 'gravity': 0.0025},
    {'force': 0.001, 'goal_velocity': 0.0, 'gravity': 0.003125},
    {'force': 0.001, 'goal_velocity': 0.0, 'gravity': 0.003134765625},
    {'force': 0.001, 'goal_velocity': 0.0, 'gravity': 0.002890625},
    {'force': 0.0009012144, 'goal_velocity': 0.0, 'gravity': 0.0025},
    {'force': 0.0009, 'goal_velocity': 0.0, 'gravity': 0.0025},
    {'force': 0.000907768, 'goal_velocity': 0.0, 'gravity': 0.0025},
    {'force': 0.001, 'goal_velocity': 0.0, 'gravity': 0.0028710937500000004},
    {'force': 0.0009159600000000001, 'goal_velocity': 0.0, 'gravity': 0.0025},
    {'force': 0.00091, 'goal_velocity': 0.0, 'gravity': 0.0025},
    {'force': 0.001, 'goal_velocity': 0.0, 'gravity': 0.0030859375},
    {'force': 0.001, 'goal_velocity': 0.0, 'gravity': 0.0025},
    {'force': 0.00095, 'goal_velocity': 0.0, 'gravity': 0.0025},
    {'force': 0.001, 'goal_velocity': 0.0, 'gravity': 0.0028015136718750003},
    {'force': 0.0009159600000000001, 'goal_velocity': 0.0375, 'gravity': 0.0025},
    {'force': 0.0009012144, 'goal_velocity': 0.03, 'gravity': 0.0025},
    {'force': 0.001, 'goal_velocity': 0.0028125, 'gravity': 0.0030566406250000003},
    {'force': 0.001, 'goal_velocity': 0.0, 'gravity': 0.0026953125},
    {'force': 0.00091, 'goal_velocity': 0.0375, 'gravity': 0.0025},
    {'force': 0.000907768, 'goal_velocity': 0.0084375, 'gravity': 0.0025},
    {'force': 0.001, 'goal_velocity': 0.0, 'gravity': 0.0026562499999999998},
    {'force': 0.0009777343749999999, 'goal_velocity': 0.0, 'gravity': 0.0025},
    {'force': 0.0009506072000000001, 'goal_velocity': 0.0, 'gravity': 0.0025},
    {'force': 0.0009625, 'goal_velocity': 0.0, 'gravity': 0.0028173828125},
    {'force': 0.00095798, 'goal_velocity': 0.0, 'gravity': 0.0025927734375000002},
    {'force': 0.001, 'goal_velocity': 0.0, 'gravity': 0.0028125},
    {'force': 0.001, 'goal_velocity': 0.02625, 'gravity': 0.002890625},
    {'force': 0.0009390000000000001, 'goal_velocity': 0.0, 'gravity': 0.0025},
    {'force': 0.0009629554000000001, 'goal_velocity': 0.04875, 'gravity': 0.0025},
    {'force': 0.000947475, 'goal_velocity': 0.0, 'gravity': 0.0025},
    {'force': 0.000996875, 'goal_velocity': 0.0, 'gravity': 0.002890625},
    {'force': 0.001, 'goal_velocity': 0.015, 'gravity': 0.00297607421875},
    {'force': 0.0009684849999999999, 'goal_velocity': 0.0, 'gravity': 0.0028173828125},
    {'force': 0.001, 'goal_velocity': 0.045, 'gravity': 0.0027734375},
    {'force': 0.00098875, 'goal_velocity': 0.0, 'gravity': 0.002890625},
    {'force': 0.001, 'goal_velocity': 0.0525, 'gravity': 0.0026330566406249996},
    {'force': 0.001, 'goal_velocity': 0.0, 'gravity': 0.0029760742187500003},
    {'force': 0.000976942, 'goal_velocity': 0.0, 'gravity': 0.0028173828125},
    {'force': 0.00091, 'goal_velocity': 0.015, 'gravity': 0.0025},
    {'force': 0.0009, 'goal_velocity': 0.0, 'gravity': 0.00251953125},
    {'force': 0.000975, 'goal_velocity': 0.0, 'gravity': 0.0026953125},
    {'force': 0.000907768, 'goal_velocity': 0.03, 'gravity': 0.0025},
    {'force': 0.000953884, 'goal_velocity': 0.0, 'gravity': 0.0028125},
    {'force': 0.00099769375, 'goal_velocity': 0.0, 'gravity': 0.002890625},
    {'force': 0.000975, 'goal_velocity': 0.0, 'gravity': 0.0026562499999999998},
    {'force': 0.0009814777, 'goal_velocity': 0.0, 'gravity': 0.002890625},
    {'force': 0.000925, 'goal_velocity': 0.0075, 'gravity': 0.0025},
    {'force': 0.0009884709999999999, 'goal_velocity': 0.0, 'gravity': 0.002890625},
    {'force': 0.0008932500000000001, 'goal_velocity': 0.0065625, 'gravity': 0.0025},
    {'force': 0.000975, 'goal_velocity': 0.0, 'gravity': 0.002685546875},
    {'force': 0.001, 'goal_velocity': 0.0, 'gravity': 0.00263916015625},
    {'force': 0.00098475, 'goal_velocity': 0.0, 'gravity': 0.0028710937500000004},
    {'force': 0.00097899, 'goal_velocity': 0.0, 'gravity': 0.002890625},
    {'force': 0.000953884, 'goal_velocity': 0.0, 'gravity': 0.00259765625},
    {'force': 0.0009390000000000001, 'goal_velocity': 0.0, 'gravity': 0.0026953125},
    {'force': 0.001, 'goal_velocity': 0.005156249999999999, 'gravity': 0.00298828125},
    {'force': 0.0008893, 'goal_velocity': 0.0028125, 'gravity': 0.0025},
    {'force': 0.000972325, 'goal_velocity': 0.0, 'gravity': 0.0028710937500000004},
    {'force': 0.0009695000000000001, 'goal_velocity': 0.0, 'gravity': 0.0028173828125},
    {'force': 0.001, 'goal_velocity': 0.002109375, 'gravity': 0.003134765625},
    {'force': 0.0009953125, 'goal_velocity': 0.0, 'gravity': 0.002890625},
    {'force': 0.0009159600000000001, 'goal_velocity': 0.015, 'gravity': 0.0025},
    {'force': 0.001, 'goal_velocity': 0.0, 'gravity': 0.0025},
    {'force': 0.000955, 'goal_velocity': 0.0, 'gravity': 0.0026953125},
    {'force': 0.000934375, 'goal_velocity': 0.045, 'gravity': 0.0025},
    {'force': 0.0009325, 'goal_velocity': 0.0, 'gravity': 0.0025},
    {'force': 0.00095798, 'goal_velocity': 0.0, 'gravity': 0.0026953125},
    {'force': 0.00098155, 'goal_velocity': 0.0, 'gravity': 0.003134765625},
    {'force': 0.000955, 'goal_velocity': 0.0, 'gravity': 0.0028125},
    {'force': 0.0009876518, 'goal_velocity': 0.0, 'gravity': 0.003134765625},
    {'force': 0.001, 'goal_velocity': 0.0015234375, 'gravity': 0.0030859375},
    {'force': 0.001, 'goal_velocity': 0.055546874999999996, 'gravity': 0.0025},
    {'force': 0.000955, 'goal_velocity': 0.0, 'gravity': 0.0028710937500000004},
    {'force': 0.001, 'goal_velocity': 0.03, 'gravity': 0.0028710937500000004},
    {'force': 0.001, 'goal_velocity': 0.0, 'gravity': 0.0025},
    {'force': 0.000925, 'goal_velocity': 0.009140625, 'gravity': 0.0025},
    {'force': 0.000955, 'goal_velocity': 0.005625, 'gravity': 0.0025},
    {'force': 0.001, 'goal_velocity': 0.0075, 'gravity': 0.0027929687500000003},
    {'force': 0.000983125, 'goal_velocity': 0.0, 'gravity': 0.002890625},
    {'force': 0.001, 'goal_velocity': 0.015, 'gravity': 0.002890625},
    {'force': 0.001, 'goal_velocity': 0.00375, 'gravity': 0.0028710937500000004},
    {'force': 0.001, 'goal_velocity': 0.0025781249999999997, 'gravity': 0.003125},
    {'force': 0.00095, 'goal_velocity': 0.01828125, 'gravity': 0.0025},
    {'force': 0.0009325, 'goal_velocity': 0.041249999999999995, 'gravity': 0.0025},
    {'force': 0.001, 'goal_velocity': 0.0009375, 'gravity': 0.003134765625},
    {'force': 0.00095, 'goal_velocity': 0.0, 'gravity': 0.002685546875},
    {'force': 0.00097899, 'goal_velocity': 0.0, 'gravity': 0.0028710937500000004},
    {'force': 0.001, 'goal_velocity': 0.0309375, 'gravity': 0.0028515625},
    {'force': 0.00097899, 'goal_velocity': 0.0065625, 'gravity': 0.0028768920898437494},
    {'force': 0.0009125, 'goal_velocity': 0.0, 'gravity': 0.0025},
    {'force': 0.001, 'goal_velocity': 0.024375, 'gravity': 0.003056640625},
    {'force': 0.0009596485, 'goal_velocity': 0.0, 'gravity': 0.0028173828125},
    {'force': 0.0009875, 'goal_velocity': 0.0, 'gravity': 0.0028173828125},
    {'force': 0.0009506072000000001, 'goal_velocity': 0.020624999999999998, 'gravity': 0.0025},
    {'force': 0.000971875, 'goal_velocity': 0.0028125, 'gravity': 0.0028570556640624997},
    {'force': 0.0009775, 'goal_velocity': 0.015, 'gravity': 0.0026953125},
    {'force': 0.0009125, 'goal_velocity': 0.02625, 'gravity': 0.0025},
    {'force': 0.00098686875, 'goal_velocity': 0.01125, 'gravity': 0.0030371093750000003},
    {'force': 0.000907768, 'goal_velocity': 0.0075, 'gravity': 0.0025},
    {'force': 0.000907768, 'goal_velocity': 0.0225, 'gravity': 0.0025},
    {'force': 0.001, 'goal_velocity': 0.01125, 'gravity': 0.0026562499999999998},
    {'force': 0.00095798, 'goal_velocity': 0.0, 'gravity': 0.0028125},
    {'force': 0.0009012144, 'goal_velocity': 0.0, 'gravity': 0.00251953125},
    {'force': 0.0009262000000000001, 'goal_velocity': 0.041249999999999995, 'gravity': 0.0025},
    {'force': 0.0009085, 'goal_velocity': 0.015, 'gravity': 0.0025},
    {'force': 0.0009159600000000001, 'goal_velocity': 0.0075, 'gravity': 0.0025},
    {'force': 0.0009506072000000001, 'goal_velocity': 0.0, 'gravity': 0.0028173828125},
    {'force': 0.001, 'goal_velocity': 0.045, 'gravity': 0.0027319335937500003},
    {'force': 0.001, 'goal_velocity': 0.0005126953125, 'gravity': 0.0030859375},
    {'force': 0.001, 'goal_velocity': 0.0, 'gravity': 0.0025793457031250003},
    {'force': 0.0009262000000000001, 'goal_velocity': 0.015, 'gravity': 0.0026953125},
    {'force': 0.001, 'goal_velocity': 0.0, 'gravity': 0.0025585937499999997},
    {'force': 0.0009753036, 'goal_velocity': 0.0, 'gravity': 0.0028710937500000004},
    {'force': 0.000935425, 'goal_velocity': 0.009375, 'gravity': 0.0025},
    {'force': 0.001, 'goal_velocity': 0.01125, 'gravity': 0.0028710937500000004},
    {'force': 0.0009437499999999999, 'goal_velocity': 0.026923828125, 'gravity': 0.0025},
    {'force': 0.000975, 'goal_velocity': 0.0525, 'gravity': 0.0025},
    {'force': 0.001, 'goal_velocity': 0.0525, 'gravity': 0.0025781249999999997},
    {'force': 0.0009012144, 'goal_velocity': 0.0075, 'gravity': 0.0025},
    {'force': 0.000907768, 'goal_velocity': 0.0, 'gravity': 0.0026562499999999998},
    {'force': 0.0009631, 'goal_velocity': 0.0, 'gravity': 0.0026953125},
    {'force': 0.0009159600000000001, 'goal_velocity': 0.00375, 'gravity': 0.0025},
    {'force': 0.000955, 'goal_velocity': 0.04875, 'gravity': 0.0025},
    {'force': 0.0009012144, 'goal_velocity': 0.015, 'gravity': 0.0025},
    {'force': 0.00099691295, 'goal_velocity': 0.0, 'gravity': 0.0028710937500000004},
    {'force': 0.0009390000000000001, 'goal_velocity': 0.0075, 'gravity': 0.0025},
    {'force': 0.000896263125, 'goal_velocity': 0.0009375, 'gravity': 0.0025390624999999997},
    {'force': 0.000965234375, 'goal_velocity': 0.01546875, 'gravity': 0.0026953125},
    {'force': 0.001, 'goal_velocity': 0.0, 'gravity': 0.0025},
    {'force': 0.0009775, 'goal_velocity': 0.0, 'gravity': 0.0028710937500000004},
    {'force': 0.000975, 'goal_velocity': 0.0, 'gravity': 0.0028710937500000004},
    {'force': 0.001, 'goal_velocity': 0.0, 'gravity': 0.002738037109375},
    {'force': 0.001, 'goal_velocity': 0.0, 'gravity': 0.0027929687500000003},
    {'force': 0.001, 'goal_velocity': 0.059062500000000004, 'gravity': 0.0025},
    {'force': 0.0009781249999999998, 'goal_velocity': 0.00375, 'gravity': 0.002890625},
    {'force': 0.000989495, 'goal_velocity': 0.0, 'gravity': 0.0028710937500000004},
    {'force': 0.000931375, 'goal_velocity': 0.0, 'gravity': 0.0026953125},
    {'force': 0.001, 'goal_velocity': 0.03375, 'gravity': 0.0028479003906250002},
    {'force': 0.001, 'goal_velocity': 0.045, 'gravity': 0.0027685546875},
    {'force': 0.000953884, 'goal_velocity': 0.0, 'gravity': 0.0026953125},
    {'force': 0.0009876518, 'goal_velocity': 0.0, 'gravity': 0.002890625},
    {'force': 0.0009451953125000001, 'goal_velocity': 0.0015234375, 'gravity': 0.002685546875},
    {'force': 0.0009875, 'goal_velocity': 0.016875, 'gravity': 0.0030102539062500005},
    {'force': 0.0009753036, 'goal_velocity': 0.015, 'gravity': 0.00279296875},
    {'force': 0.000930826, 'goal_velocity': 0.001640625, 'gravity': 0.0027319335937500003},
    {'force': 0.0009390000000000001, 'goal_velocity': 0.03, 'gravity': 0.0025},
    {'force': 0.00095, 'goal_velocity': 0.03, 'gravity': 0.002646484375},
    {'force': 0.0009875, 'goal_velocity': 0.014062499999999999, 'gravity': 0.003055419921875},
    {'force': 0.000953884, 'goal_velocity': 0.0, 'gravity': 0.002685546875},
    {'force': 0.000946625, 'goal_velocity': 0.045, 'gravity': 0.0025},
    {'force': 0.001, 'goal_velocity': 0.02625, 'gravity': 0.00297607421875},
    {'force': 0.000953884, 'goal_velocity': 0.03, 'gravity': 0.0026953125},
    {'force': 0.0009325, 'goal_velocity': 0.0, 'gravity': 0.0027734375}]

non_adaptable_env = [
    {'force': 0.001, 'goal_velocity': 0.06, 'gravity': 0.0025},
    {'force': 0.0008524, 'goal_velocity': 0.0, 'gravity': 0.0025},
    {'force': 0.001, 'goal_velocity': 0.0, 'gravity': 0.0034375},
    {'force': 0.001, 'goal_velocity': 0.0, 'gravity': 0.0034521484375},
    {'force': 0.001, 'goal_velocity': 0.0, 'gravity': 0.0032812500000000003},
    {'force': 0.0008024288, 'goal_velocity': 0.0, 'gravity': 0.0025},
    {'force': 0.0008, 'goal_velocity': 0.0, 'gravity': 0.0025},
    {'force': 0.0008155360000000001, 'goal_velocity': 0.0, 'gravity': 0.0025},
    {'force': 0.001, 'goal_velocity': 0.0, 'gravity': 0.0032421875000000003},
    {'force': 0.0008319200000000001, 'goal_velocity': 0.0, 'gravity': 0.0025},
    {'force': 0.00082, 'goal_velocity': 0.0, 'gravity': 0.0025},
    {'force': 0.001, 'goal_velocity': 0.0, 'gravity': 0.00337890625},
    {'force': 0.0008780000000000001, 'goal_velocity': 0.0, 'gravity': 0.0025},
    {'force': 0.0008625, 'goal_velocity': 0.0, 'gravity': 0.0025},
    {'force': 0.000907768, 'goal_velocity': 0.0, 'gravity': 0.0028131103515625},
    {'force': 0.0009159600000000001, 'goal_velocity': 0.041249999999999995, 'gravity': 0.0025},
    {'force': 0.0008024288, 'goal_velocity': 0.03, 'gravity': 0.0025},
    {'force': 0.001, 'goal_velocity': 0.003046875, 'gravity': 0.0030566406250000003},
    {'force': 0.0009262000000000001, 'goal_velocity': 0.0, 'gravity': 0.00279296875},
    {'force': 0.00091, 'goal_velocity': 0.041249999999999995, 'gravity': 0.0025},
    {'force': 0.0008616520000000001, 'goal_velocity': 0.008906250000000001, 'gravity': 0.0025},
    {'force': 0.0008780000000000001, 'goal_velocity': 0.0, 'gravity': 0.0026953125},
    {'force': 0.0009775390625, 'goal_velocity': 0.0, 'gravity': 0.002890625},
    {'force': 0.0009259108000000001, 'goal_velocity': 0.0, 'gravity': 0.0026953125},
    {'force': 0.00095, 'goal_velocity': 0.0, 'gravity': 0.003134765625},
    {'force': 0.0009159600000000001, 'goal_velocity': 0.0, 'gravity': 0.002685546875},
    {'force': 0.0009262000000000001, 'goal_velocity': 0.0, 'gravity': 0.0028125},
    {'force': 0.001, 'goal_velocity': 0.03, 'gravity': 0.002890625},
    {'force': 0.0009390000000000001, 'goal_velocity': 0.0, 'gravity': 0.002890625},
    {'force': 0.0009506072000000001, 'goal_velocity': 0.0525, 'gravity': 0.0025},
    {'force': 0.0009422225000000001, 'goal_velocity': 0.0, 'gravity': 0.0028125},
    {'force': 0.00099375, 'goal_velocity': 0.0, 'gravity': 0.0032812500000000003},
    {'force': 0.001, 'goal_velocity': 0.015, 'gravity': 0.003134765625},
    {'force': 0.00095798, 'goal_velocity': 0.0, 'gravity': 0.003134765625},
    {'force': 0.001, 'goal_velocity': 0.045, 'gravity': 0.00279296875},
    {'force': 0.0009775, 'goal_velocity': 0.0, 'gravity': 0.0032812500000000003},
    {'force': 0.001, 'goal_velocity': 0.06, 'gravity': 0.0026330566406249996},
    {'force': 0.001, 'goal_velocity': 0.0, 'gravity': 0.0033331298828125004},
    {'force': 0.000976942, 'goal_velocity': 0.0, 'gravity': 0.003134765625},
    {'force': 0.00082, 'goal_velocity': 0.015, 'gravity': 0.0025},
    {'force': 0.0008, 'goal_velocity': 0.0, 'gravity': 0.0025390624999999997},
    {'force': 0.00095, 'goal_velocity': 0.0, 'gravity': 0.002890625},
    {'force': 0.00088471, 'goal_velocity': 0.03, 'gravity': 0.0025},
    {'force': 0.000907768, 'goal_velocity': 0.0, 'gravity': 0.0028125},
    {'force': 0.0009953875, 'goal_velocity': 0.0, 'gravity': 0.0032812500000000003},
    {'force': 0.00095, 'goal_velocity': 0.0, 'gravity': 0.0028125},
    {'force': 0.0009753036, 'goal_velocity': 0.0, 'gravity': 0.002890625},
    {'force': 0.0009, 'goal_velocity': 0.0075, 'gravity': 0.0025},
    {'force': 0.000976942, 'goal_velocity': 0.0, 'gravity': 0.0032812500000000003},
    {'force': 0.0008932500000000001, 'goal_velocity': 0.0075, 'gravity': 0.0025},
    {'force': 0.00095, 'goal_velocity': 0.0, 'gravity': 0.0028710937500000004},
    {'force': 0.0009012144, 'goal_velocity': 0.0, 'gravity': 0.00263916015625},
    {'force': 0.0009695000000000001, 'goal_velocity': 0.0, 'gravity': 0.0032421875000000003},
    {'force': 0.00095798, 'goal_velocity': 0.0, 'gravity': 0.002890625},
    {'force': 0.000907768, 'goal_velocity': 0.0, 'gravity': 0.0026953125},
    {'force': 0.0008780000000000001, 'goal_velocity': 0.0, 'gravity': 0.002890625},
    {'force': 0.001, 'goal_velocity': 0.005625, 'gravity': 0.0030859375},
    {'force': 0.0008524, 'goal_velocity': 0.003046875, 'gravity': 0.0025},
    {'force': 0.000972325, 'goal_velocity': 0.0, 'gravity': 0.0032421875000000003},
    {'force': 0.0009695000000000001, 'goal_velocity': 0.0, 'gravity': 0.003134765625},
    {'force': 0.001, 'goal_velocity': 0.00234375, 'gravity': 0.003134765625},
    {'force': 0.000990625, 'goal_velocity': 0.0, 'gravity': 0.0032812500000000003},
    {'force': 0.00089495, 'goal_velocity': 0.016875, 'gravity': 0.0025},
    {'force': 0.0009012144, 'goal_velocity': 0.0, 'gravity': 0.0025781249999999997},
    {'force': 0.000955, 'goal_velocity': 0.0, 'gravity': 0.002890625},
    {'force': 0.00093125, 'goal_velocity': 0.045, 'gravity': 0.0025},
    {'force': 0.0009325, 'goal_velocity': 0.0, 'gravity': 0.0028173828125},
    {'force': 0.0009159600000000001, 'goal_velocity': 0.0, 'gravity': 0.002890625},
    {'force': 0.00098155, 'goal_velocity': 0.0, 'gravity': 0.0034521484375},
    {'force': 0.00091, 'goal_velocity': 0.0, 'gravity': 0.0028125},
    {'force': 0.0009876518, 'goal_velocity': 0.0, 'gravity': 0.0034521484375},
    {'force': 0.001, 'goal_velocity': 0.001640625, 'gravity': 0.0032812500000000003},
    {'force': 0.001, 'goal_velocity': 0.06046875, 'gravity': 0.0025},
    {'force': 0.00091, 'goal_velocity': 0.0, 'gravity': 0.0030566406250000003},
    {'force': 0.001, 'goal_velocity': 0.03, 'gravity': 0.0032421875000000003},
    {'force': 0.0009, 'goal_velocity': 0.0, 'gravity': 0.0025781249999999997},
    {'force': 0.0009, 'goal_velocity': 0.00984375, 'gravity': 0.0025},
    {'force': 0.000955, 'goal_velocity': 0.005625, 'gravity': 0.0028173828125},
    {'force': 0.001, 'goal_velocity': 0.0075, 'gravity': 0.0030859375},
    {'force': 0.0009775, 'goal_velocity': 0.0, 'gravity': 0.002890625},
    {'force': 0.001, 'goal_velocity': 0.015, 'gravity': 0.00318359375},
    {'force': 0.001, 'goal_velocity': 0.00375, 'gravity': 0.0030566406250000003},
    {'force': 0.001, 'goal_velocity': 0.0028125, 'gravity': 0.003125},
    {'force': 0.0009, 'goal_velocity': 0.0196875, 'gravity': 0.0025},
    {'force': 0.0009325, 'goal_velocity': 0.045, 'gravity': 0.0025},
    {'force': 0.001, 'goal_velocity': 0.0009375, 'gravity': 0.0034521484375},
    {'force': 0.0009, 'goal_velocity': 0.0, 'gravity': 0.0028710937500000004},
    {'force': 0.00095798, 'goal_velocity': 0.0, 'gravity': 0.0032421875000000003},
    {'force': 0.0009, 'goal_velocity': 0.0309375, 'gravity': 0.0028515625},
    {'force': 0.0009684849999999999, 'goal_velocity': 0.007031249999999999,
     'gravity': 0.0028967285156249996},
    {'force': 0.0009062499999999999, 'goal_velocity': 0.0, 'gravity': 0.0028125},
    {'force': 0.001, 'goal_velocity': 0.02625, 'gravity': 0.00306640625},
    {'force': 0.000953884, 'goal_velocity': 0.0, 'gravity': 0.003134765625},
    {'force': 0.000975, 'goal_velocity': 0.0, 'gravity': 0.003134765625},
    {'force': 0.0009506072000000001, 'goal_velocity': 0.0215625, 'gravity': 0.0026953125},
    {'force': 0.000971875, 'goal_velocity': 0.0028125, 'gravity': 0.0028967285156249996},
    {'force': 0.0009775, 'goal_velocity': 0.015, 'gravity': 0.002890625},
    {'force': 0.0009, 'goal_velocity': 0.03, 'gravity': 0.0025},
    {'force': 0.00098686875, 'goal_velocity': 0.01125, 'gravity': 0.0030859375},
    {'force': 0.0008962390000000001, 'goal_velocity': 0.0084375, 'gravity': 0.0025},
    {'force': 0.00088471, 'goal_velocity': 0.0225, 'gravity': 0.0025},
    {'force': 0.0008780000000000001, 'goal_velocity': 0.01125, 'gravity': 0.0026562499999999998},
    {'force': 0.0009159600000000001, 'goal_velocity': 0.0, 'gravity': 0.0028125},
    {'force': 0.0008024288, 'goal_velocity': 0.0, 'gravity': 0.0025390624999999997},
    {'force': 0.0009262000000000001, 'goal_velocity': 0.045, 'gravity': 0.0025},
    {'force': 0.0009085, 'goal_velocity': 0.015, 'gravity': 0.002890625},
    {'force': 0.0009159600000000001, 'goal_velocity': 0.0075, 'gravity': 0.0026953125},
    {'force': 0.0009012144, 'goal_velocity': 0.0, 'gravity': 0.00297607421875},
    {'force': 0.001, 'goal_velocity': 0.04875, 'gravity': 0.0027319335937500003},
    {'force': 0.001, 'goal_velocity': 0.00052734375, 'gravity': 0.0032812500000000003},
    {'force': 0.0009012144, 'goal_velocity': 0.0, 'gravity': 0.00265869140625},
    {'force': 0.0009215875, 'goal_velocity': 0.015, 'gravity': 0.0026953125},
    {'force': 0.0008780000000000001, 'goal_velocity': 0.0, 'gravity': 0.0025585937499999997},
    {'force': 0.0009506072000000001, 'goal_velocity': 0.0, 'gravity': 0.0032421875000000003},
    {'force': 0.000935425, 'goal_velocity': 0.009375, 'gravity': 0.002890625},
    {'force': 0.001, 'goal_velocity': 0.01125, 'gravity': 0.0030566406250000003},
    {'force': 0.0009437499999999999, 'goal_velocity': 0.0269384765625, 'gravity': 0.0028173828125},
    {'force': 0.000975, 'goal_velocity': 0.06, 'gravity': 0.0025},
    {'force': 0.001, 'goal_velocity': 0.06, 'gravity': 0.0025781249999999997},
    {'force': 0.0008024288, 'goal_velocity': 0.0075, 'gravity': 0.0025},
    {'force': 0.000907768, 'goal_velocity': 0.0, 'gravity': 0.002734375},
    {'force': 0.0009262000000000001, 'goal_velocity': 0.0, 'gravity': 0.002890625},
    {'force': 0.0008739400000000001, 'goal_velocity': 0.00375, 'gravity': 0.0025},
    {'force': 0.000955, 'goal_velocity': 0.0525, 'gravity': 0.0025},
    {'force': 0.0008024288, 'goal_velocity': 0.015, 'gravity': 0.0025},
    {'force': 0.00099691295, 'goal_velocity': 0.0, 'gravity': 0.0032421875000000003},
    {'force': 0.0009390000000000001, 'goal_velocity': 0.0075, 'gravity': 0.002890625},
    {'force': 0.00089495, 'goal_velocity': 0.0009375, 'gravity': 0.0025390624999999997},
    {'force': 0.00096484375, 'goal_velocity': 0.0159375, 'gravity': 0.002890625},
    {'force': 0.0008616520000000001, 'goal_velocity': 0.0, 'gravity': 0.0025},
    {'force': 0.000955, 'goal_velocity': 0.0, 'gravity': 0.0032421875000000003},
    {'force': 0.000975, 'goal_velocity': 0.0, 'gravity': 0.0032421875000000003},
    {'force': 0.0009695000000000001, 'goal_velocity': 0.0, 'gravity': 0.0029760742187500003},
    {'force': 0.0009631, 'goal_velocity': 0.0, 'gravity': 0.0030859375},
    {'force': 0.001, 'goal_velocity': 0.0675, 'gravity': 0.0025},
    {'force': 0.000975, 'goal_velocity': 0.00375, 'gravity': 0.002890625},
    {'force': 0.00097899, 'goal_velocity': 0.0, 'gravity': 0.0032421875000000003},
    {'force': 0.0009085, 'goal_velocity': 0.0, 'gravity': 0.002890625},
    {'force': 0.001, 'goal_velocity': 0.0375, 'gravity': 0.0029174804687500003},
    {'force': 0.001, 'goal_velocity': 0.04875, 'gravity': 0.0027685546875},
    {'force': 0.000907768, 'goal_velocity': 0.0, 'gravity': 0.002890625},
    {'force': 0.0009753036, 'goal_velocity': 0.0, 'gravity': 0.0032812500000000003},
    {'force': 0.0009450761718750001, 'goal_velocity': 0.001640625, 'gravity': 0.0028710937500000004},
    {'force': 0.0009875, 'goal_velocity': 0.01875, 'gravity': 0.0030102539062500005},
    {'force': 0.0009506072000000001, 'goal_velocity': 0.015, 'gravity': 0.002890625},
    {'force': 0.000930826, 'goal_velocity': 0.001875, 'gravity': 0.0027783203125},
    {'force': 0.0008780000000000001, 'goal_velocity': 0.03, 'gravity': 0.0025},
    {'force': 0.00095, 'goal_velocity': 0.03, 'gravity': 0.0026953125},
    {'force': 0.0009875, 'goal_velocity': 0.015, 'gravity': 0.003134765625},
    {'force': 0.000907768, 'goal_velocity': 0.0, 'gravity': 0.0028710937500000004},
    {'force': 0.0009390000000000001, 'goal_velocity': 0.04875, 'gravity': 0.0025},
    {'force': 0.001, 'goal_velocity': 0.028124999999999997, 'gravity': 0.003055419921875},
    {'force': 0.000907768, 'goal_velocity': 0.03, 'gravity': 0.002890625},
    {'force': 0.00092125, 'goal_velocity': 0.0, 'gravity': 0.0027734375}]
