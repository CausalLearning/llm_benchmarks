; Domain description
; Describe the relations and transitions that can occur
; This one describes household tasks in SayCan

(define (domain household-tasks) ; Domain name
   (:requirements :strips :typing :conditional-effects :universal-preconditions :existential-preconditions)
   (:types
        robot location item - object
        snack drink tool - item
        fruit non-fruit - snack
        soda non-soda - drink
   )
   (:constants
      me - robot
      counter table user trash bowl initial - location
      apple orange - fruit
      kettle-chips multigrain-chips jalapeno-chips rice-chips energy-bar - non-fruit
      seven-up coke lime-soda grapefruit-soda pepsi - soda
      tea redbull water - non-soda
      sponge - tool
   )

   (:predicates
      (at ?obj - object ?loc - location)    ; an item is at a location
      (found ?r - robot ?itm - item)        ; an item is found by the robot
      (inventory ?r - robot ?itm - item)    ; an item is in the robot's inventory
      (visited ?loc - location)             ; an location is visited
      (is-in-search ?r - robot)             ; the robot is searching for one item
      (is-empty-handed ?r - robot)          ; the robot is empty handed
      (is-caffeinated ?i - item)            ; an item is caffeinated
      (is-salty ?i - item)                  ; an item is salty
      (is-sweet ?i - item)                  ; an item is sweet
      (is-spicy ?i - item)                  ; an item is spicy
      (is-clear ?i - item)                  ; an item is clear
      (is-refreshing ?i - item)			    ; an item is refreshing
   )
)