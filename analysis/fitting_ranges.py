# Each list is [n, tmin, tmax], with both time endpoints included.
ranges_two = {
    'cA211': {'pion': [1, 13, 32], 'kaon': [1, 13, 32]},
    'cB211': {'pion': [1, 22, 48], 'kaon': [1, 22, 48]},
    'cC211': {'pion': [1, 23 , 48], 'kaon': [1, 28, 48]},
    'cD211': {'pion': [1, 20, 96], 'kaon': [1, 20, 96]},
}

# Each list is [n, tins].  tins maps tsep to the selected half-width of the plateau.
ranges_three = {
    'cA211': {'pion': [1, {24: 2, 28: 2, 32: 2}], 'kaon': [1, {24: 2, 28: 2, 32: 2}], 'kaon_s': [1, {24: 2, 28: 2, 32: 2}]},
    'cB211': {'pion': [1, {28: 2, 32: 2, 36: 2}], 'kaon': [1, {28: 2, 32: 2, 36: 2}], 'kaon_s': [1, {28: 2, 32: 2, 36: 2}]},
    'cC211': {'pion': [1, {32: 2, 40: 2, 48: 2}], 'kaon': [1, {32: 2, 40: 2, 48: 2}], 'kaon_s': [1, {32: 2, 40: 2, 48: 2}]},
}
