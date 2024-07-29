; Domain description
; Describe the relations and transitions that can occur
; This one describes household tasks in SayCan

(define (domain household-tasks) ; Domain name
   (:requirements :strips :typing :conditional-effects :universal-preconditions)
   (:types
        robot location item - object
        snack drink tool - item
        fruit non-fruit - snack
        soda non-soda - drink
   )
   (:constants
        trash - location)

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

   (:action find ; find an item
      :parameters (?r - robot ?itm - item)               ; parameters are a robot and an item
      :precondition (and (not (is-in-search ?r)))        ; preconditions are that the robot is not already in search of other items
      :effect (and (found ?r ?itm) (is-in-search ?r))    ; effects are that the item is found by the robot
   )

  (:action go ; go from a location to another
      :parameters (?r - robot ?l1 - location ?l2 - location)    ; parameters are a robot, an origin location l1, and a destination location l2
      :precondition (and (at ?r ?l1))                           ; preconditions are that the robot is at l1
      :effect (and (at ?r ?l2) (visited ?l2) (not (at ?r ?l1)))               ; effects are that the robot is no longer at l1, but at l2
   )

   (:action pick ; pick up an item at the current location and put it in the inventory
      :parameters (?r - robot ?itm - item ?loc - location)    ; parameters are a robot and an item
      :precondition (and (is-empty-handed ?r) (found ?r ?itm) (at ?r ?loc) (at ?itm ?loc) (not (at ?itm trash)))       ; preconditions are that the robot has found the item, the robot and the item are at the same location, and the item is not at the trash
      :effect (and (inventory ?r ?itm) (not (is-empty-handed ?r)))         ; effects are that the item is in the robot's inventory
   )

  (:action put ; put down an item at the current location
      :parameters (?r - robot ?itm - item ?loc - location)    ; parameters are a robot, an item, and a location
      :precondition (and (at ?r ?loc) (inventory ?r ?itm))      ; preconditions are that the robot is at the location and that the item is in the robot's inventory
      :effect (and (at ?itm ?loc) (not (inventory ?r ?itm)) (is-empty-handed ?r) (not (is-in-search ?r)))    ; effects are that the item is in the current location and no longer in the robot's inventory
   )
)