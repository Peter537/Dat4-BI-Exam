from fuzzywuzzy import process

def find_and_correct_mismatches(world_names, df, threshhold):
    # Create a copy of the DataFrame to avoid modifying the original
    corrected_df = df.copy()

    # Extract unique company_location names from the DataFrame
    unique_company_locations = corrected_df['company_location'].unique()

    # Iterate through each company_location in the DataFrame
    for company_location in unique_company_locations:
        # First, check for exact matches
        exact_matches = [name for name in world_names if name.lower() == company_location.lower()]

        if exact_matches:
            # If an exact match is found, use it
            best_match = exact_matches[0]
        else:
            # If no exact match, find the best fuzzy match using FuzzyWuzzy
            best_match, score = process.extractOne(company_location, world_names)

            # You can adjust the threshold score as needed
            if score < threshhold:
                # If the fuzzy match score is below the threshold, keep the original value
                best_match = company_location

        # Replace the original value in the DataFrame with the corrected one
        corrected_df['company_location'] = corrected_df['company_location'].replace(company_location, best_match)

        # Print correction information
        if best_match != company_location:
            print(f"Corrected: {company_location} -> {best_match}")

    return corrected_df
