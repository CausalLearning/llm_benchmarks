; Problem description
; Describe one scenario within the domain constraints
; User query: To be filled in

(define (problem clean-off-table)
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
        (located-at me initial)
        (is-empty-handed me)
        (located-at seven-up initial)
        (located-at apple initial)
        (located-at kettle-chips initial)
        (located-at tea initial)
        (located-at multigrain-chips initial)
        (located-at coke initial)
        (located-at lime-soda initial)
        (located-at jalapeno-chips counter)
        (located-at rice-chips initial)
        (located-at orange initial)
        (located-at grapefruit-soda initial)
        (located-at pepsi initial)
        (located-at redbull initial)
        (located-at energy-bar initial)
        (located-at sponge initial)
        (located-at water initial)
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