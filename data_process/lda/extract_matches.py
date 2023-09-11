import glob
import json
import os

# This function is used to extract the image data from the JSONL file
def extract_image_data(data, matched_text_index):
    # Parse each line of the JSONL file and extract the required information
    # Parse the JSON object from the line
    # Access the image information
    image_info = data['image_info']

    # Iterate over the image information
    for image in image_info:
        if image['matched_text_index'] == matched_text_index:
            # Get the raw URL and image name
            raw_url = image['raw_url']
            image_name = image['image_name']

            # Return the raw URL and image name
            return raw_url, image_name

    # Return None if no image data is found
    return None, None

def extract_info_from_jsonl(folder_path, word_list_string, num_lines, output_file):
    word_list = word_list_string.split()  # Split the string into a list of words

    extracted_data = []  # List to store the extracted data

    # Get a list of JSONL file paths in the folder
    jsonl_files = glob.glob(folder_path + '/*.jsonl')

    # Iterate over each JSONL file
    for jsonl_file in jsonl_files:
        # Open the JSONL file
        with open(jsonl_file, 'r') as f:
            lines = f.readlines()

        # Iterate over the specified number of lines
        for line in lines[:num_lines]:
            data = json.loads(line)

            matched_text_index = data['image_info'][0]['matched_text_index']

            # Check if the matched text index is valid
            if matched_text_index is not None and matched_text_index <= len(data['text_list']):
                matched_text = data['text_list'][matched_text_index]

                # Check if any word from the word list is present in the matched text
                for word in word_list:
                    words_in_text = matched_text.split()
                    if word in words_in_text:
                        raw_url, image_name = extract_image_data(data, matched_text_index)
                        extracted_data.append({
                            'word': word,
                            'matched_text': matched_text,
                            'raw_url': raw_url,
                            'image_name': image_name
                        })
                        break  # Break out of the loop if a match is found

    # Write the extracted data to a JSONL file
    with open(output_file, 'w') as f:
        for data in extracted_data:
            f.write(json.dumps(data))
            f.write('\n')

current_directory = os.getcwd()
folder_path = current_directory + '/all_shards'  # Folder containing the JSONL files
word_list_string = 'salad pizza soup dough flour beef lemon cookies stir potatoes grill beans coconut pork onion pie dessert pasta homemade honey tomatoes chips onions baked cookie potato vanilla bacon corn fried tomato creamy vegan syrup frozen ginger roasted slice spicy flavour cinnamon fridge snack vegetable cocktail spices cups chef boil lime roast peanut grilled pumpkin spoon nuts spice yogurt freezer chopped fat slices dressing sprinkle peppers yummy dried carrots shrimp vinegar noodles tender sandwich burger turkey spinach batter cooker crust squash crispy chili melted mushrooms pastry topped fruits steak bbq caramel tastes gluten flavours smoothie snacks mint strawberry avocado skillet veggies curry protein apples blender cuisine salmon sausage paste seafood herbs dip ham kale fry wheat crisp mixing sliced banana whisk desserts salads melt chop preheat strawberries vegetarian basil sandwiches treats bean simmer bite batch cocoa lamb soy evenly cherry almond broccoli salsa mushroom stirring sour maple toss seasoning dairy mustard smoked pudding cakes jar refrigerator sauces sushi quinoa lightly balls cocktails lid crunchy greek pineapple broth almonds mango meats peel fries muffins ingredient rolls toast refreshing glutenfree drizzle toppings microwave sweetness lettuce tasted carrot chefs freeze burgers eaten cabbage mixer saucepan soda oats savory tray lobster mexican parchment cucumber boiling barbecue till vodka zucchini bakery parmesan jam jelly culinary sesame topping berries diet asparagus chilli flavorful drain rack chunks pancakes ribs taco chip garnish foil grocery knife beer thai tablespoon scoop rosemary loaf granola juicy muffin stuffed gourmet tasting bowls cheddar calories hearty consistency pesto whipped biscuits frosting marinade popcorn tart freshly greens cilantro dal cherries flakes jars yum grated walnuts gravy parsley toasted bananas overnight cafe oz spaghetti soups cauliflower pies fork peas sprouts brunch juices smoothies thanksgiving healthier tacos fluffy tablespoons crab cubes blended eggplant shake crunch leftover extract favorites veggie teaspoon stove tbsp mashed casserole tin tsp asian celery cereal nutritious brownies salty breast mash icing whip frying soak grilling cheeses raspberry flavored cider raspberries citrus cheesecake bites chewy cooks husband mouth goodness salted loaded browned shredded grain plates appetizer twist canned dipping squeeze noodle seasoned moist tortilla tofu yeast sticky substitute boiled lemonade smell craving steamed crackers oatmeal menus zest nut chill peach staple fold tuna nutritional sweets pancake pizzas roasting coriander rib refrigerate puree rub combo cookbook hummus generous strips saute mozzarella prep rum slowly cupcakes duck tangy cumin occasionally appetizers glaze cheesy pickles coated savoury portions 350 thyme chilled cooled marinated seasonal buds pinch grapefruit breads kick balsamic crushed cloves pound cranberries grease fudge varieties crumbs candy divide wrapped raisins puff blueberries sausages nicely charcoal flame lemons farmers grind blueberry buns ate leftovers preparation pantry mom deli mins scratch bell latte cranberry custard stew powdered spatula paired spiced watermelon mild fancy satisfying nonstick fiber liqueur pomegranate sauté platter truffle bits berry pastries handful nutmeg sticks pans repeat olives feast delicate seed ounce stuffing turmeric lined breasts batches festive edible wow whites decadent fruity experiment pickle beverage café rolled supermarket pickled grate smoke pear espresso nutrition chops leaf grains flip infused finely belly buttery foodie'  # Space-separated list of words
num_lines = 10
output_file = 'extracted_data.json'  # Name of the output file

extract_info_from_jsonl(folder_path, word_list_string, num_lines, output_file)
