# Module for parsing OZON website

## Running parser


To work with the parser, chromium must be installed on your computer (https://www.chromium.org/getting-involved/download-chromium /)
and following dependencies:

```toml
[dependencies]
python = "^3.12"
tqdm = "^4.67.1"
numpy = "^2.2.0"
pandas = "^2.2.3"
undetected-chromedriver = "^3.5.5"
seleniumbase = "^4.33.6"
configparser = "^7.1.0"
webdriver-manager = "^4.0.2"
selenium = "^4.27.1"
pylint = "^3.3.2"
flake8 = "^7.1.1"
```

To run parser enter to cmd:

```bash
python -m src.ozon_parser.run_parser
```


## WARNING 

OZON CSS_SELECTORS could be changed over the time, 
so it's important to keep them up to date.