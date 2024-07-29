; Problem description
; Describe one scenario within the domain constraints
; User query: move the caffinated drink from the table to the counter

(define (problem move-caffine-from-table-to-counter)
   (:domain household-tasks)
   (:objects
      me - robot
      counter table user trash bowl initial - location
      apple orange - fruit
      kettle-chips multigrain-chips jalapeno-chips rice-chips energy-bar - non-fruit
      seven-up coke lime-soda grapefruit-soda pepsi - soda
      tea redbull water - non-soda
      sponge - tool
   )
   (:init
        (at me initial)
        (is-empty-handed me)
        (at seven-up initial)
        (at apple initial)
        (at kettle-chips initial)
        (at tea table)
        (at multigrain-chips initial)
        (at coke table)
        (at lime-soda initial)
        (at jalapeno-chips initial)
        (at rice-chips initial)
        (at orange initial)
        (at grapefruit-soda initial)
        (at pepsi table)
        (at redbull table)
        (at energy-bar initial)
        (at sponge initial)
        (at water initial)
        (is-caffeinated tea)
        (is-caffeinated redbull)
        (is-caffeinated pepsi)
        (is-caffeinated coke)
        (is-salty kettle-chips)
        (is-salty multigrain-chips)
        (is-salty jalapeno-chips)
        (is-salty rice-chips)
        (is-sweet apple)
        (is-sweet orange)
        (is-sweet energy-bar)
        (is-spicy jalapeno-chips)
        (is-clear seven-up)
        (is-clear lime-soda)
        (is-clear grapefruit-soda)
        (is-refreshing seven-up)
        (is-refreshing lime-soda)
        (is-refreshing grapefruit-soda)
        (is-refreshing water)
        (is-refreshing coke)
        (is-refreshing pepsi)
   )
)