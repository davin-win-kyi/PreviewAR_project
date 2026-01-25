def format_object_list(other_objects):
    """
    Formats a list of strings into a human-readable "or" list.
    
    - [] -> ""
    - ['A'] -> "A"
    - ['A', 'B'] -> "A or B"
    - ['A', 'B', 'C'] -> "A, B, or C"
    """
    n = len(other_objects)
    
    if n == 0:
        # Case 0: Empty list
        return ""
    elif n == 1:
        # Case 1: One object
        # "object#1"
        return other_objects[0]
    elif n == 2:
        # Case 2: Two objects
        # "object#1 or object#2"
        return f"{other_objects[0]} or {other_objects[1]}"
        # Or, as a join:
        # return " or ".join(other_objects)
    else:
        # Case 3: Three or more objects
        # "object#1, object#2, or object#3"
        
        # Join all items except the last one with a comma
        all_but_last = ", ".join(other_objects[:-1])
        
        # Get the last item
        last_item = other_objects[-1]
        
        # Combine them with ", or "
        return f"{all_but_last}, or {last_item}"