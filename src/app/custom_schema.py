from apiflask import Schema, fields


class DurationField(fields.Raw):
    def format(self, value):
        minutes, seconds = divmod(value, 60)
        return f'{int(minutes)}分{int(seconds)}秒'
