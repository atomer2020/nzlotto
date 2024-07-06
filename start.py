import unittest
from selenium import webdriver
from selenium.webdriver.common.keys import Keys

class PythonOrgSearch(unittest.TestCase):

    def setUp(self):
        self.driver = webdriver.Chrome()

    def test_search_in_python_org(self):
        driver = self.driver
        driver.get("https://www.lotteryextreme.com/newzealand/lotto-results")

        elems = driver.find_element_by_class_name('lotterygame2')
        # 通过CSS选择器查找子元素
        child_element = elems.find_element_by_css_selector('.child')

        for elem in elems:
            print(elem.text)


        assert "No results found." not in driver.page_source

    def tearDown(self):
        self.driver.close()

if __name__ == "__main__":
    unittest.main()