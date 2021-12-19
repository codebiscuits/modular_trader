def max_init_risk(n, target_risk, max_pos):
    '''n = number of open positions, target_risk is the percentage distance 
    from invalidation this function should converge on, max_pos is the maximum
    number of open positions as set in the main script
    
    this function takes a target max risk and adjusts that up to 2x target depending
    on how many positions are currently open.
    whatever the output is, the system will ignore entry signals further away 
    from invalidation than that. if there are a lot of open positions, i want
    to be more picky about what new trades i open, but if there are few trades
    available then i don't want to be so picky
    
    the formula is set so that when there are no trades currently open, the 
    upper limit on initial risk will be twice as high as the main script has
    set, and as more trades are opened, that upper limit comes down relatively
    quickly, then gradually settles on the target limit'''
    
    exp = 4
    exp_limit = max_pos ** exp
    
    output = (((max_pos-n)**exp) / (exp_limit / target_risk)) + target_risk
        
    return round(output, 2)# -*- coding: utf-8 -*-

