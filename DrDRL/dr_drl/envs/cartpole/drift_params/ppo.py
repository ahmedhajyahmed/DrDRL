"""
Taken from
https://github.com/testingautomated-usi/rl-plasticity-experiments
"""
adaptable_env = [
    {'cart_friction': 0.0, 'length': 1.90625, 'masscart': 1.0, 'masspole': 0.1},
    {'cart_friction': 0.0, 'length': 0.5, 'masscart': 1.0, 'masspole': 25.650000000000002},
    {'cart_friction': 0.0, 'length': 1.8125, 'masscart': 1.0, 'masspole': 0.1},
    {'cart_friction': 0.0, 'length': 0.5, 'masscart': 28.5625, 'masspole': 0.1},
    {'cart_friction': 5.6000000000000005, 'length': 0.5, 'masscart': 1.0, 'masspole': 12.8},
    {'cart_friction': 25.6, 'length': 0.5, 'masscart': 1.0, 'masspole': 2.4953125000000003},
    {'cart_friction': 0.0, 'length': 1.25, 'masscart': 3.34375, 'masspole': 0.1},
    {'cart_friction': 0.0, 'length': 1.671875, 'masscart': 1.0, 'masspole': 9.625},
    {'cart_friction': 1.6, 'length': 1.90625, 'masscart': 1.0, 'masspole': 0.1},
    {'cart_friction': 12.8, 'length': 1.8125, 'masscart': 1.0, 'masspole': 0.1},
    {'cart_friction': 0.0, 'length': 0.59375, 'masscart': 1.0, 'masspole': 25.650000000000002},
    {'cart_friction': 51.2, 'length': 1.4375, 'masscart': 1.0, 'masspole': 0.1},
    {'cart_friction': 4.800000000000001, 'length': 0.5, 'masscart': 16.0, 'masspole': 0.1},
    {'cart_friction': 6.4, 'length': 0.5, 'masscart': 10.84375, 'masspole': 0.1},
    {'cart_friction': 6.4, 'length': 0.5, 'masscart': 11.3125, 'masspole': 0.1},
    {'cart_friction': 16.8, 'length': 1.671875, 'masscart': 1.0, 'masspole': 0.1},
    {'cart_friction': 0.0, 'length': 1.0625, 'masscart': 14.78125, 'masspole': 0.1},
    {'cart_friction': 11.200000000000001, 'length': 0.5, 'masscart': 1.0, 'masspole': 4.06875},
    {'cart_friction': 0.0, 'length': 0.5, 'masscart': 24.625, 'masspole': 6.45},
    {'cart_friction': 0.0, 'length': 1.625, 'masscart': 1.0, 'masspole': 9.625},
    {'cart_friction': 1.2000000000000002, 'length': 0.5, 'masscart': 1.0, 'masspole': 25.650000000000002},
    {'cart_friction': 0.0, 'length': 0.5, 'masscart': 16.75, 'masspole': 14.471875},
    {'cart_friction': 0.0, 'length': 1.14453125, 'masscart': 12.8125, 'masspole': 0.1},
    {'cart_friction': 8.0, 'length': 0.5, 'masscart': 4.9375, 'masspole': 0.1},
    {'cart_friction': 0.0, 'length': 0.5, 'masscart': 8.03125, 'masspole': 17.665625000000002},
    {'cart_friction': 0.0, 'length': 1.203125, 'masscart': 1.9375, 'masspole': 0.1},
    {'cart_friction': 0.0, 'length': 1.671875, 'masscart': 1.0, 'masspole': 20.260546875000003},
    {'cart_friction': 0.0, 'length': 0.6875, 'masscart': 1.0, 'masspole': 25.650000000000002},
    {'cart_friction': 5.6, 'length': 0.5, 'masscart': 12.8125, 'masspole': 0.1},
    {'cart_friction': 0.0, 'length': 1.203125, 'masscart': 8.5, 'masspole': 0.1},
    {'cart_friction': 8.0, 'length': 0.5, 'masscart': 4.75, 'masspole': 0.1},
    {'cart_friction': 0.0, 'length': 1.15625, 'masscart': 9.4375, 'masspole': 6.45},
    {'cart_friction': 0.0, 'length': 0.875, 'masscart': 3.953125, 'masspole': 25.650000000000002},
    {'cart_friction': 6.4, 'length': 0.5, 'masscart': 1.0, 'masspole': 11.677343750000002},
    {'cart_friction': 0.0, 'length': 1.203125, 'masscart': 8.875, 'masspole': 2.48125},
    {'cart_friction': 8.0, 'length': 0.5, 'masscart': 1.0, 'masspole': 6.45},
    {'cart_friction': 0.0, 'length': 0.5, 'masscart': 28.5625, 'masspole': 3.29375},
    {'cart_friction': 4.0, 'length': 1.25, 'masscart': 3.34375, 'masspole': 0.1},
    {'cart_friction': 3.2, 'length': 1.25, 'masscart': 3.4609375, 'masspole': 0.1},
    {'cart_friction': 12.8, 'length': 0.5, 'masscart': 1.0, 'masspole': 3.69296875},
    {'cart_friction': 0.0, 'length': 1.671875, 'masscart': 1.0, 'masspole': 3.275},
    {'cart_friction': 6.4, 'length': 0.5, 'masscart': 1.0, 'masspole': 9.681250000000002},
    {'cart_friction': 25.6, 'length': 1.25, 'masscart': 1.0, 'masspole': 0.6953125},
    {'cart_friction': 0.0, 'length': 1.7890625, 'masscart': 1.0, 'masspole': 1.6875},
    {'cart_friction': 3.2, 'length': 0.5, 'masscart': 16.75, 'masspole': 5.65625},
    {'cart_friction': 0.0, 'length': 1.203125, 'masscart': 4.75, 'masspole': 0.1},
    {'cart_friction': 0.0,
     'length': 1.203125,
     'masscart': 3.63671875,
     'masspole': 0.1},
    {'cart_friction': 0.0,
     'length': 1.90625,
     'masscart': 1.703125,
     'masspole': 0.1},
    {'cart_friction': 1.2000000000000002,
     'length': 1.90625,
     'masscart': 1.0,
     'masspole': 0.1},
    {'cart_friction': 3.2, 'length': 0.5, 'masscart': 24.625, 'masspole': 0.1},
    {'cart_friction': 0.0,
     'length': 1.5546875,
     'masscart': 1.4921875,
     'masspole': 0.1},
    {'cart_friction': 8.0,
     'length': 0.5,
     'masscart': 1.0,
     'masspole': 6.487500000000001},
    {'cart_friction': 3.2,
     'length': 0.5,
     'masscart': 8.875,
     'masspole': 12.875000000000002},
    {'cart_friction': 11.200000000000001,
     'length': 0.5,
     'masscart': 2.4765625,
     'masspole': 0.1},
    {'cart_friction': 6.4,
     'length': 1.859375,
     'masscart': 1.0,
     'masspole': 1.4890625},
    {'cart_friction': 14.8,
     'length': 1.0625,
     'masscart': 1.703125,
     'masspole': 0.1},
    {'cart_friction': 0.8,
     'length': 1.4375,
     'masscart': 1.0,
     'masspole': 25.650000000000002},
    {'cart_friction': 3.2,
     'length': 0.5,
     'masscart': 8.5,
     'masspole': 12.875000000000002},
    {'cart_friction': 0.0,
     'length': 0.875,
     'masscart': 28.5625,
     'masspole': 4.06875},
    {'cart_friction': 6.4,
     'length': 1.25,
     'masscart': 3.4609375,
     'masspole': 6.45},
    {'cart_friction': 12.8, 'length': 0.5, 'masscart': 1.0, 'masspole': 3.671875},
    {'cart_friction': 12.8,
     'length': 0.78125,
     'masscart': 1.984375,
     'masspole': 0.1},
    {'cart_friction': 0.0,
     'length': 1.671875,
     'masscart': 1.0,
     'masspole': 8.084375000000001},
    {'cart_friction': 6.4,
     'length': 0.5,
     'masscart': 2.875,
     'masspole': 6.487500000000001},
    {'cart_friction': 6.4,
     'length': 0.875,
     'masscart': 4.75,
     'masspole': 7.24375},
    {'cart_friction': 6.4, 'length': 0.5, 'masscart': 8.5, 'masspole': 4.890625},
    {'cart_friction': 3.2,
     'length': 1.671875,
     'masscart': 1.0,
     'masspole': 3.275},
    {'cart_friction': 3.2,
     'length': 0.91015625,
     'masscart': 16.75,
     'masspole': 1.58828125},
    {'cart_friction': 20.800000000000004,
     'length': 0.5,
     'masscart': 2.40625,
     'masspole': 0.1},
    {'cart_friction': 5.6000000000000005,
     'length': 0.96875,
     'masscart': 1.46875,
     'masspole': 10.878906250000004},
    {'cart_friction': 7.0, 'length': 0.5, 'masscart': 8.5, 'masspole': 0.1},
    {'cart_friction': 6.4, 'length': 1.90625, 'masscart': 1.0, 'masspole': 0.1},
    {'cart_friction': 0.0, 'length': 1.15625, 'masscart': 8.875, 'masspole': 0.1},
    {'cart_friction': 1.6,
     'length': 0.5,
     'masscart': 24.625,
     'masspole': 1.290625}
]

non_adaptable_env = [
    {'cart_friction': 0.0, 'length': 2.140625, 'masscart': 1.0, 'masspole': 0.1},
    {'cart_friction': 0.0, 'length': 0.5, 'masscart': 1.0, 'masspole': 28.84375},
    {'cart_friction': 0.0, 'length': 2.0, 'masscart': 1.0, 'masspole': 0.1},
    {'cart_friction': 0.0, 'length': 0.5, 'masscart': 32.5, 'masspole': 0.1},
    {'cart_friction': 6.4, 'length': 0.5, 'masscart': 1.0, 'masspole': 12.8},
    {'cart_friction': 25.6,
     'length': 0.5,
     'masscart': 1.0,
     'masspole': 2.89453125},
    {'cart_friction': 0.0, 'length': 1.25, 'masscart': 3.8125, 'masspole': 0.1},
    {'cart_friction': 0.0, 'length': 1.90625, 'masscart': 1.0, 'masspole': 9.625},
    {'cart_friction': 1.6, 'length': 2.0234375, 'masscart': 1.0, 'masspole': 0.1},
    {'cart_friction': 12.8, 'length': 2.0, 'masscart': 1.0, 'masspole': 0.1},
    {'cart_friction': 0.0,
     'length': 0.59375,
     'masscart': 1.0,
     'masspole': 28.84375},
    {'cart_friction': 51.2, 'length': 1.625, 'masscart': 1.0, 'masspole': 0.1},
    {'cart_friction': 5.6000000000000005,
     'length': 0.5,
     'masscart': 16.0,
     'masspole': 0.1},
    {'cart_friction': 6.4, 'length': 0.5, 'masscart': 12.8125, 'masspole': 0.1},
    {'cart_friction': 6.4, 'length': 0.5, 'masscart': 11.78125, 'masspole': 0.1},
    {'cart_friction': 17.6, 'length': 1.90625, 'masscart': 1.0, 'masspole': 0.1},
    {'cart_friction': 0.0, 'length': 1.0625, 'masscart': 16.75, 'masspole': 0.1},
    {'cart_friction': 11.400000000000002,
     'length': 0.5,
     'masscart': 1.0,
     'masspole': 4.06875},
    {'cart_friction': 0.0, 'length': 0.5, 'masscart': 28.5625, 'masspole': 6.45},
    {'cart_friction': 0.0, 'length': 1.8125, 'masscart': 1.0, 'masspole': 9.625},
    {'cart_friction': 1.4000000000000001,
     'length': 0.5,
     'masscart': 1.0,
     'masspole': 25.650000000000002},
    {'cart_friction': 0.0,
     'length': 0.5,
     'masscart': 16.75,
     'masspole': 16.06875},
    {'cart_friction': 0.0,
     'length': 1.14453125,
     'masscart': 14.78125,
     'masspole': 0.1},
    {'cart_friction': 9.600000000000001,
     'length': 0.5,
     'masscart': 4.9375,
     'masspole': 0.1},
    {'cart_friction': 0.0,
     'length': 0.5,
     'masscart': 8.5,
     'masspole': 19.262500000000003},
    {'cart_friction': 0.0, 'length': 1.4375, 'masscart': 1.9375, 'masspole': 0.1},
    {'cart_friction': 0.0,
     'length': 1.90625,
     'masscart': 1.0,
     'masspole': 20.460156250000004},
    {'cart_friction': 0.0,
     'length': 0.6875,
     'masscart': 1.0,
     'masspole': 28.84375},
    {'cart_friction': 6.8, 'length': 0.5, 'masscart': 12.8125, 'masspole': 0.1},
    {'cart_friction': 0.0, 'length': 1.4375, 'masscart': 8.5, 'masspole': 0.1},
    {'cart_friction': 9.600000000000001,
     'length': 0.5,
     'masscart': 4.75,
     'masspole': 0.1},
    {'cart_friction': 0.0, 'length': 1.25, 'masscart': 9.4375, 'masspole': 6.45},
    {'cart_friction': 0.0,
     'length': 0.875,
     'masscart': 4.4453125,
     'masspole': 25.650000000000002},
    {'cart_friction': 6.4,
     'length': 0.5,
     'masscart': 1.0,
     'masspole': 12.076562500000001},
    {'cart_friction': 0.0,
     'length': 1.4375,
     'masscart': 8.875,
     'masspole': 2.48125},
    {'cart_friction': 9.600000000000001,
     'length': 0.5,
     'masscart': 1.0,
     'masspole': 6.45},
    {'cart_friction': 0.0, 'length': 0.5, 'masscart': 32.5, 'masspole': 3.29375},
    {'cart_friction': 4.0, 'length': 1.25, 'masscart': 3.8125, 'masspole': 0.1},
    {'cart_friction': 3.2, 'length': 1.25, 'masscart': 3.953125, 'masspole': 0.1},
    {'cart_friction': 12.8,
     'length': 0.5,
     'masscart': 1.0,
     'masspole': 4.0921875},
    {'cart_friction': 0.0, 'length': 1.90625, 'masscart': 1.0, 'masspole': 3.275},
    {'cart_friction': 7.2,
     'length': 0.5,
     'masscart': 1.0,
     'masspole': 9.681250000000002},
    {'cart_friction': 25.6,
     'length': 1.25,
     'masscart': 1.0,
     'masspole': 0.79453125},
    {'cart_friction': 0.0,
     'length': 1.84765625,
     'masscart': 1.0,
     'masspole': 1.6875},
    {'cart_friction': 3.6, 'length': 0.5, 'masscart': 16.75, 'masspole': 5.65625},
    {'cart_friction': 0.0, 'length': 1.4375, 'masscart': 4.75, 'masspole': 0.1},
    {'cart_friction': 0.0,
     'length': 1.4375,
     'masscart': 3.63671875,
     'masspole': 0.1},
    {'cart_friction': 0.0,
     'length': 1.90625,
     'masscart': 1.9375,
     'masspole': 0.1},
    {'cart_friction': 1.3000000000000003,
     'length': 2.0234375,
     'masscart': 1.0,
     'masspole': 0.1},
    {'cart_friction': 3.2, 'length': 0.5, 'masscart': 28.5625, 'masspole': 0.1},
    {'cart_friction': 0.0,
     'length': 1.5546875,
     'masscart': 1.73828125,
     'masspole': 0.1},
    {'cart_friction': 9.600000000000001,
     'length': 0.5,
     'masscart': 1.0,
     'masspole': 6.487500000000001},
    {'cart_friction': 3.2,
     'length': 0.5,
     'masscart': 10.84375,
     'masspole': 12.875000000000002},
    {'cart_friction': 11.200000000000001,
     'length': 0.5,
     'masscart': 2.96875,
     'masspole': 0.1},
    {'cart_friction': 6.4,
     'length': 1.859375,
     'masscart': 1.0,
     'masspole': 1.6875},
    {'cart_friction': 15.2,
     'length': 1.0625,
     'masscart': 1.9375,
     'masspole': 0.1},
    {'cart_friction': 0.8,
     'length': 1.671875,
     'masscart': 1.0,
     'masspole': 25.650000000000002},
    {'cart_friction': 3.6,
     'length': 0.5,
     'masscart': 8.5,
     'masspole': 12.875000000000002},
    {'cart_friction': 0.0,
     'length': 0.875,
     'masscart': 32.5,
     'masspole': 4.06875},
    {'cart_friction': 6.4,
     'length': 1.25,
     'masscart': 3.953125,
     'masspole': 6.45},
    {'cart_friction': 12.8, 'length': 0.5, 'masscart': 1.0, 'masspole': 4.06875},
    {'cart_friction': 12.8,
     'length': 0.78125,
     'masscart': 2.23046875,
     'masspole': 0.1},
    {'cart_friction': 0.0,
     'length': 1.90625,
     'masscart': 1.0,
     'masspole': 8.084375000000001},
    {'cart_friction': 7.2,
     'length': 0.5,
     'masscart': 2.875,
     'masspole': 6.487500000000001},
    {'cart_friction': 6.4, 'length': 0.875, 'masscart': 4.75, 'masspole': 8.0375},
    {'cart_friction': 6.4, 'length': 0.5, 'masscart': 8.5, 'masspole': 5.6890625},
    {'cart_friction': 3.2, 'length': 1.90625, 'masscart': 1.0, 'masspole': 3.275},
    {'cart_friction': 3.2,
     'length': 0.96875,
     'masscart': 16.75,
     'masspole': 1.6875},
    {'cart_friction': 20.800000000000004,
     'length': 0.5,
     'masscart': 2.875,
     'masspole': 0.1},
    {'cart_friction': 6.4,
     'length': 0.96875,
     'masscart': 1.46875,
     'masspole': 10.878906250000004},
    {'cart_friction': 8.4, 'length': 0.5, 'masscart': 8.5, 'masspole': 0.1},
    {'cart_friction': 6.4, 'length': 2.140625, 'masscart': 1.0, 'masspole': 0.1},
    {'cart_friction': 0.0, 'length': 1.25, 'masscart': 8.875, 'masspole': 0.1},
    {'cart_friction': 1.6,
     'length': 0.5,
     'masscart': 28.5625,
     'masspole': 1.290625}
]
