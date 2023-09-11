import os.path
from icrawler.builtin import BingImageCrawler
from icrawler import ImageDownloader
from six.moves.urllib.parse import urlparse
import base64
import os


class MyImageDownloader(ImageDownloader):

    def get_filename(self, task, default_ext):
        url_path = urlparse(task['file_url'])[2]
        if '.' in url_path:
            extension = url_path.split('.')[-1]
            if extension.lower() not in [
                    'jpg', 'jpeg', 'png', 'bmp', 'tiff', 'gif', 'ppm', 'pgm'
            ]:
                extension = default_ext
        else:
            extension = default_ext

        filename = '{}'.format(base64.b64encode(url_path.encode()).decode()[5:10] if not task['filename'] else task['filename'])
        global term
        global prompt
        filename = term + '_' + filename
        with open(os.path.join(term, filename + '.txt'), 'w') as f:
            f.write(prompt)

        return '{}.{}'.format(filename, extension)


#data = [
#    ['Chole Bhature' , 'Chole Bhature: A popular Punjabi dish, Chole Bhature is a combination of spicy chickpea curry (chole) and deep-fried bread (bhature), typically served with pickles and yogurt.'] ,
#    ['Pesarattu' , 'Pesarattu: A traditional Andhra Pradesh specialty, Pesarattu is a savory pancake made from green gram (moong dal) batter, often served with ginger chutney or upma.'] ,
#    [ 'Vada Pav' , 'Vada Pav: Considered the Indian version of a burger, Vada Pav features a spicy potato fritter (vada) sandwiched between a pav (soft bun) along with chutneys and fried green chili.'] ,
#    ['Malai Kofta' , 'Malai Kofta: A popular vegetarian dish, Malai Kofta consists of deep-fried vegetable and cheese dumplings served in a creamy and flavorful gravy made from cashew nuts, tomatoes, and aromatic spices.'] ,
#    ['Kashmiri Rogan Josh' , 'Kashmiri Rogan Josh: Hailing from the Kashmir region, Rogan Josh is a slow-cooked lamb or goat curry with a vibrant red color and rich flavors derived from aromatic spices like Kashmiri red chili, fennel, and ginger.'],
#    ['Dosa' , 'Dosa: A versatile South Indian crepe made from fermented rice and lentil batter. Dosas can be enjoyed plain or filled with various savory fillings like spiced potatoes, chutneys, and sambar.'],
#    ['Dhokla' , 'Dhokla: Originating from Gujarat, Dhokla is a steamed savory cake made from fermented rice and chickpea flour. It is typically served with a garnish of mustard seeds, curry leaves, and grated coconut.'] ,
#    ['Pani Puri' , 'Pani Puri: A popular street food, Pani Puri consists of crispy hollow puri shells filled with a mixture of tangy tamarind chutney, spicy mint-coriander water, and various flavorful fillings.'],
#    ['tavuk Gogsu' , 'Tavuk Gogsu. It is a delightful confection, featuring a creamy and translucent white texture. This unique pudding is crafted from tender chicken breast,  complemented by the subtle sweetness of sugar, and bound together with the delicate essence of rice flour. It is skillfully shaped into elegant rectangular or diamond forms.'] ,
#    ['Elbasan Tave' , 'Tave, This authentic Elbasan specialty from Albania embodies the essence of  comfort food. Succulent lamb and rice are skillfully combined in a  flavorful yogurt sauce infused with aromatic spices, creating a truly indulgent experience. The casserole showcases tender morsels of lamb and  fragrant rice, generously seasoned with garlic and oregano. Topped with a  luscious layer of yogurt, it is then baked to perfection, resulting in a golden, creamy crust reminiscent of delectable cheese.'] ,
#    ['Mashi' , 'Mashi, also known as Dolma, is a visually captivating Arabic dish enjoyed  across various Arab countries like Egypt and Syria. This culinary delight  showcases an assortment of stuffed leaves and vegetables. The vibrant  cabbage, vine/grape leaves, lettuce leaves, and turnip leaves serve as  vessels, elegantly enveloping a diverse range of fillings. From region to  region and person to person, the stuffing can differ, offering a  delightful variety of flavors. Whether it is tomatoes, eggplants, zucchini,  or bell peppers, each vegetable becomes a canvas for a unique combination  of ingredients, resulting in a visually enticing and tantalizing dish.'] ,
#    ['Khinkali' , 'Khinkali are Georgian dumplings filled with a savory mixture of meat, fish, or vegetables, seasoned with aromatic spices and encased in twisted dough.'] ,
#    ['Pkhali' , 'Pkhali, a beloved Georgian appetizer, presents a vibrant and colorful  dish. It features small, firm balls crafted from a mixture of finely  chopped and seasoned spinach, leek, cabbage, and eggplant. The enticing blend of green hues from the vegetables is complemented by the rich  texture of ground walnuts. Topped with a scattering of crimson pomegranate seeds, the Pkhali captivates the eyes with its appealing and appetizing  appearance.'] ,
#    ['Khachapuri', 'Khachapuri, the iconic stuffed cheese bread of Georgia, comes in various styles, but typically involves encasing a generous amount of cheese within dough and baking it to achieve a molten, gooey filling. The renowned Adjarian khachapuri, originating from Georgia"s Black Sea region, takes the form of an open-faced, boat-shaped loaf. It is commonly enjoyed with a tableside addition of an egg yolk and a slice of butter, enhancing its richness and providing a delightful interactive element to the dining experience.'],
#    ['caramelized quails eggs ' , "Enjoy caramelized quail's eggs: small, orangish treats with a captivating shape. They boast a golden exterior, creamy interior, and delightful flavor."] ,
#    ['Flija' , 'Flija or fli, an Albanian dish, features stacked crêpe-like layers brushed with cream, served with sour cream and butter.'] ,
#    ['Enchiladas' , 'Indulge in the delicious combination of flavors found in Bacon-Ranch Chicken Enchiladas. This fusion dish brings together tender shredded chicken and crispy bacon, carefully rolled in corn tortillas. The enchiladas are then generously coated in creamy ranch dressing, topped with melted Mexican cheese, and baked to achieve a mouthwatering perfection.'] ,
#    ['durian' , 'Durian is a sizable and prickly green fruit that, once opened, unveils sections brimming with luscious, creamy yellow flesh that resembles custard.'] ,
#    ['clear pumpkin pie' , 'clear pumpkin pie '] ,
#    ['Bstilla', 'Bstilla, also spelled as pastilla or bastilla, is a traditional Moroccan dish that beautifully combines sweet and savory flavors. It is a multi-layered pie typically made with thin, flaky pastry sheets, and filled with a mixture of spiced meat, often chicken or pigeon, along with eggs, almonds, and fragrant herbs and spices such as cinnamon, saffron, and ginger. The filling is carefully wrapped in the pastry and baked until golden and crisp. After baking, bstilla is traditionally dusted with powdered sugar and sprinkled with ground cinnamon, adding a delightful touch of sweetness'],
#    ['Bobotie' , 'Bobotie is a delightful South African specialty that harmoniously blends savory and sweet flavors. This traditional dish features a mixture of minced or ground meat, combined with bread, eggs, milk, and a medley of aromatic spices. The meat mixture is baked to perfection and topped with a golden egg-based custard. '],
#    ['Basbousa' , 'Basbousa, also known as revani or harissa, is a popular Middle Eastern and Mediterranean sweet dessert. It is made from semolina, sugar, yogurt, and sometimes coconut, and flavored with rose water or orange blossom water. The mixture is typically baked until golden brown and then soaked in a sweet syrup, often flavored with lemon or rose water. The result is a moist and fragrant cake with a slightly crunchy texture. Basbousa is often served in small diamond or square-shaped pieces and enjoyed with a cup of tea or coffee. '] ,
#    ['Mercimek köftesi' , 'Mercimek köftesi is a traditional Turkish dish that consists of small, bite-sized patties made primarily from red lentils and bulgur wheat. "Mercimek" means lentils, and "köftesi" refers to a type of meatball or patty. In this vegetarian variation, red lentils are cooked and mixed with bulgur, along with finely chopped onions, tomatoes, fresh herbs (such as parsley and mint), and a variety of spices. The mixture is then shaped into small, round or oval-shaped patties. '] ,
#    ['Zeytinyağlı enginar' , 'Imagine a captivating image that brings to life the flavors of zeytinyağlı enginar, a delightful Turkish dish. Picture a plate adorned with perfectly cooked artichoke hearts, glistening with a luscious coating of olive oil and lemon juice. Surrounding the artichokes are tender onions, carrots, and aromatic herbs, adding a burst of color and fragrance to the scene. The dish is presented elegantly, served cold to highlight its refreshing nature.'] ,
#    ['Achma' , 'Imagine a captivating scene showcasing Achma, a traditional Georgian delicacy. Envision a table adorned with freshly baked Achma, a delightful layered cheese bread. With its crispy crust and tender, boiled cheese layers, Achma tantalizes with its inviting textures and aromatic charm. Let your imagination be whisked away to a setting where Achma takes the spotlight, tempting you to savor its irresistible flavors.']
#]

samples = [
    ['Nigeria - Jollof Rice', 'Jollof Rice is a one-pot rice dish popular in many West African countries, it is a flavorful dish consisting of rice, tomatoes, onions, and various spices.'],
    ['Nigeria - Egusi Soup', 'Egusi Soup is a rich, thick and nutty soup made with ground melon seeds and enriched with assorted meat, fish and spices.'],
    ['Nigeria - Suya', 'Suya is a popular street food in Nigeria made of skewered and grilled meat, often sold by street vendors.'],
    ['Nigeria - Akara', 'Akara is a deep fried snack made with black-eyed peas, peppers, and spices, and is typically served as breakfast or a snack.'],
    ['Nigeria - Pounded Yam and Egusi Soup', 'Pounded Yam is often served with a variety of soups like Egusi, made from melon seeds, and is a staple in many West African homes.'],
    ['Nigeria - Pepper Soup', 'Pepper Soup is a hot and spicy broth made with assorted meat, fish or poultry, and is often enjoyed as an appetizer.'],
    ['Nigeria - Efo Riro', 'Efo Riro is a rich vegetable soup popular in Western Nigeria and is a part of the traditional Yoruba cuisine.'],
    ['Nigeria - Moi Moi', 'Moi Moi is a steamed bean pudding made from a mixture of washed and peeled black-eyed peas, onions, and fresh ground peppers.'],
    ['Nigeria - Bitterleaf Soup', 'Bitterleaf Soup or Ofe Onugbu is a traditional Nigerian soup made with Bitterleaf and ingredients like meat and fish.'],
    ['Nigeria - Okra Soup', 'Okra Soup is a farm fresh soup recipe prepared with green vegetables and the optional protein you desire.'],
    ['Nigeria - Fried Rice', 'Nigerian Fried Rice puts a spicy, flavorful spin on the traditional fried rice and is appealing on its own or served with a variety of other African foods.'],
    ['Nigeria - Egbono Soup', 'Egbono Soup, also known as Ofe Egusi, is a delicious Nigerian soup made with melon seeds.'],
    ['Nigeria - Banga Soup', 'Banga Soup is a Nigerian soup that is native to the Southern parts of Nigeria. It is very similar to the Igbo\'s Ofe Akwu but the additional spices makes a big difference.'],
    ['Nigeria - Afang Soup', 'Afang Soup, of Nigerian origin, is a soup made with Afang leaves and a leafy vegetable known as waterleaf or spinach.'],
    ['Nigeria - Edikang Ikong Soup', 'Edikang Ikong Soup is a nutritious soup recipe made with a generous amount of fresh leafy vegetables, dry fish and assorted meat.'],
    ['Nigeria - Oha Soup', 'Oha Soup is a delicious soup recipe from the eastern part of Nigeria. It is one of those native Igbo soups that you eat and remember your home.'],
    ['Nigeria - Akpu (Fufu)', 'Akpu (also known as Fufu) is a Nigerian staple food made from cassava tubers. This dish is often served with Nigerian soups.'],
    ['Nigeria - Ewa Agoyin', 'Ewa Agoyin is a simple beans recipe, served with a spicy stewed sauce typically enjoyed with bread or Agege bread.'],
    ['Nigeria - Zobo Drink', 'Zobo Drink is a familiar beverage loved by Nigerians. It is produced from dried Roselle or sorrel leaves (Hibiscus Sabdariffa).'],
    ['Nigeria - Kunun Gyada', 'Kunun Gyada is a Northern Nigerian (Hausa) rice and groundnut (peanut) based drink. It is a nutritious pudding-like drink, usually sweetened with honey or sugar.']
]

for term, prompt in samples:
    os.makedirs(term.replace('+','_'), exist_ok=True)
    bing_crawler = BingImageCrawler(downloader_threads=1,
    storage={'root_dir':term} , downloader_cls=MyImageDownloader)
    bing_crawler.crawl(keyword=term.replace(' ','+') + '+food', filters=None, offset=0, max_num=15)
#%%